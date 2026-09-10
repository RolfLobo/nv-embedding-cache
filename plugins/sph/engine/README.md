# `sph::ShardedPerfectHashMap`

A GPU hash map for `int64_t` keys and fixed-size opaque byte payloads, built on a
**two-level (sharded) perfect hash**. Keys are bucketized by `hash(seed, key) % num_buckets`;
each bucket stores one 8-byte **PDE** (a *pilot* plus the page it owns), and the pilot is
searched at insert time so that every key in that bucket maps to a distinct slot inside the
bucket's **PTE** run — i.e. lookup is collision-free by construction. A lookup is therefore
two dependent loads (PDE → PTE) plus one payload read, with no probing. Buckets are grouped
into *mega-shards* whose PTE and payload storage are separate VMM-backed buffers that are
committed on demand, so the table grows without reallocating or moving what is already
there. Each PTE carries a 56-bit fingerprint (`key / num_buckets`) so misses are rejected
without storing the key. 

Public header: [include/sph/sph_hash_map.h](include/sph/sph_hash_map.h),
errors: [include/sph/error.h](include/sph/error.h),
geometry constants: [include/sph/sph_geometry.h](include/sph/sph_geometry.h).

## API

All key/value/output pointers are **device** pointers. Values are
opaque `int8_t[value_size]` rows.

| Call | What it does |
|---|---|
| `ShardedPerfectHashMap(initial_capacity, value_size = 512, default_value = nullptr)` | Sizes the geometry for `initial_capacity` keys and `value_size` bytes/row. `default_value` is a host-or-device row of `value_size` bytes written for misses in `find()`; `nullptr` means all-zero. Both sizes must be `> 0`. Synchronizes before returning. |
| `insert(d_keys, d_values, num_keys, stream, on_existing)` | Incremental insert of a batch. `d_values` may be `nullptr` (rows get `default_value`); otherwise it is `num_keys * value_size` contiguous bytes — **no stride**. `on_existing` picks what happens to keys already in the table or repeated inside the batch: `ON_EXISTING_IGNORE` (default, keep the stored value), `ON_EXISTING_UPDATE` (overwrite in place; last occurrence wins), `ON_EXISTING_FAIL` (no filtering — a repeat makes the pilot search unsatisfiable and the whole bucket is dropped). Throws `SphInsertFailed` if any key was dropped. **Synchronizes** before returning. |
| `build(d_keys, num_keys, stream, d_values = nullptr)` | Bulk single-shot construction, faster than `insert` for a full population. Does **no** dedup and **no** existing-key filtering; it writes every bucket's PDE from scratch, so it is for a fresh/`reset()` table only. Throws `SphInsertFailed` on dropped keys. **Synchronizes** before returning. |
| `find(keys, num_keys, values, stream, hit_mask = nullptr, value_stride = 0)` | Gathers `num_keys` rows into `values`. `value_stride` is the byte spacing of output rows; `0` means tightly packed (`value_size`), and it must be `>= value_size`. `hit_mask` is an optional in/out bitmask of `ceil(num_keys/64)` `uint64` words: a set input bit **skips** that key, and found keys have their bit set on return. Misses get `default_value`. Async on `stream`. |
| `find_locations(d_keys, num_keys, d_locations, stream)` | Per-key byte offset of the payload row relative to `get_value_payload_ptr()`, or `-1` on miss. Doubles as a `contains`. Async on `stream`. |
| `assign(keys, values, num_keys, stream)` | Write-if-exists: overwrites the payload of keys already resident, silently ignores misses. Never grows the table. Async on `stream`. |
| `reserve(num_keys, stream = 0)` | Pre-commits VMM pages for roughly `num_keys` keys so later inserts skip the per-insert grow. Purely an optimization — an estimate, never a limit. **Synchronizes**. |
| `reset(stream, capacity)` | Re-initializes to an empty table of `capacity`. Rebuilds every buffer; `get_size()` goes to 0. |
| `defrag(mega_shard, stream)` / `defrag_all(stream)` | Compacts the payload by relocating live entries into the sibling mini-shard and releasing the source shard's VMM. **Synchronizes**. |
| `get_size()` | Live keys currently in the table. |
| `get_target_capacity()` | Capacity the geometry was sized for. A **target**, not a limit (see below). |
| `get_num_buckets()` / `get_num_shards()` / `get_global_seed()` / `get_value_size()` / `get_value_payload_ptr()` | Geometry / storage accessors, host-side and free. |
| `get_total_payload_slots()` / `get_num_ptes()` / `get_bits_per_key()` / `get_fragmentation()` | Diagnostics. `get_fragmentation()` is `payload_slots / live_keys` (≥ 1.0); it is the defrag trigger. |

## Concurrency contract

The class does **no locking of its own**; the user enforces all of it 

* **modify vs modify — mutually exclusive.** `insert` / `build` / `assign` / `reserve` /
  `reset` / `defrag*` must never overlap each other, on the same map, on any stream. The
  scratch pools, pinned readback buffers and failure counters are single-instance state and
  two concurrent modifies will corrupt them.
* **read vs insert — allowed to overlap.** `find` / `find_locations` may run concurrently
  with an in-flight `insert`.
* **read vs defrag — mutually exclusive.** `defrag` moves the payload and unmaps the source
  shard's VMM, so a concurrent `find` reads unmapped memory.
* **Diagnostics vs modify — mutually exclusive for `get_num_ptes()` only.** `get_num_ptes()`
  (and `get_bits_per_key()`, which is built on it) read device state with a blocking copy on
  the **default stream**, which does not order against a caller's `cudaStreamNonBlocking`
  stream. They report the table as of the last *completed* modify — exact right after
  `insert`/`build`/`reserve`/`defrag` return, torn if they overlap one.
  `get_total_payload_slots()` and `get_fragmentation()` read host-side commit bookkeeping
  only — no device access, no stream involvement — so they are safe to call from any stream
  context, though they still only reflect the last completed modify.
* **Streams.** `find` / `find_locations` / `assign` are asynchronous on the supplied stream
  and do not synchronize. `insert` / `build` / `reserve` / `defrag*` and the constructor
  **do** synchronize before returning, because they read back device counters on the host.

## Handling insert / build failures

`insert()` and `build()` throw `sph::SphInsertFailed` when any key was dropped. The
exception carries an `InsertResult`:

```cpp
try {
    map.insert(d_keys, d_values, n, stream);
} catch (const sph::SphInsertFailed& e) {
    const sph::InsertResult& r = e.result();
    // r.keys_submitted, r.keys_dropped, r.keys_existing, r.keys_batch_duplicates
    // r.keys_inserted()  == submitted - dropped - existing - batch_duplicates
    // r.failures.pilot_failure / max_bucket_reached
    //          / bucket_offset_overflow / page_index_overflow   (one bump per bucket)
}
```

The failure modes, all of which mean *this bucket could not be made perfect or ran out of
room*: `pilot_failure` (no pilot separates the bucket's keys — the usual one, and what a
duplicate key under `ON_EXISTING_FAIL` produces), `max_bucket_reached` (bucket exceeded the
fused kernel's gather capacity), `bucket_offset_overflow` / `page_index_overflow` (a shard's
PTE or page counter saturated its bit field).

**Insert is not atomic.** By the time the exception is thrown the rest of the batch has
already been applied and `get_size()` already reflects it. Only the buckets counted in
`failures` lost their batch keys; keys previously resident in those buckets are untouched.

**Current recovery: rebuild.** There is no per-key retry and no partial repair path. The
supported recovery is to reconstruct the table from the authoritative key set the caller
owns:

```cpp
map.reset(stream, new_capacity);   // new_capacity >= all_keys.size(), typically larger
map.build(d_all_keys, all_n, stream, d_all_values);
```

Because dropping is almost always a *density* symptom, the rebuild should raise the capacity
(or, if fragmentation is the issue rather than density, `defrag_all()` first and retry the
batch before falling back to the full rebuild). This is deliberately naive; the caller must
therefore keep its own copy of the live key/value set if it needs to survive a failed insert.

## Other contracts worth knowing

* **Capacity is a target, not a limit.** `get_size()` may exceed `get_target_capacity()`;
  the cost is a rising rate of `pilot_failure` drops. Capacity is also **floored** at
  `min_num_buckets * avg_bucket_size` (= 2048 keys): the 56-bit fingerprint is
  `key / num_buckets` and truncates below 256 buckets, which would produce wrong hits and
  corrupt `defrag`. `get_target_capacity()` reports the value actually used, not the
  requested one.
* **Values are contiguous on input.** `insert` / `build` / `assign` assume
  `stride == value_size`. Only `find` accepts an output stride. Strided callers must compact
  first (`cudaMemcpy2DAsync`).
* **`build` assumes an empty table.** It rewrites PDEs from scratch and does not dedup;
  duplicate keys in its input make that bucket unsatisfiable and drop it. Use `insert` for
  anything incremental.
* **`assign` is not an upsert.** Missing keys are silently skipped, not inserted, and not
  reported.
* **Miss detection.** There is no `contains()`. Use `find`'s `hit_mask`, or
  `find_locations` and test for `-1`. `find` overwrites miss rows with `default_value`, so a
  sentinel-filled output buffer also works.
* **Empty batches** (`num_keys == 0`) are a no-op on every entry point; negative counts
  throw `SphInvalidArgument`.
* **Payload pointer is not stable.** `get_value_payload_ptr()` and any offset from
  `find_locations` are invalidated by anything that grows or moves storage — `insert`,
  `build`, `defrag*`, `reset`.
* **Exceptions.** Everything derives from the `sph::SphException` marker base, so a single
  `catch (const sph::SphException&)` covers the engine. Individually:
  `SphInvalidArgument : std::invalid_argument` (bad argument / precondition),
  `SphOutOfMemory : std::bad_alloc` (allocation, including VMM commit),
  `SphCudaError : SphError : std::runtime_error` (any other CUDA/driver status, `code()`
  carries the raw `cudaError_t`/`CUresult`), `SphInsertFailed : SphError` (dropped keys).
* **Device affinity.** The map binds to whatever CUDA device was current at construction;
  the caller must set the same device before every call.