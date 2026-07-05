# Layer Pooling & Dequantization Test Suite

`layer_pooling_test` is a unified, capability-driven suite that validates embedding-layer
**pooling** and **dequantization** across **all** layer types against a single reference
(`MockHostTable::combine()`).

Each generated case drives a real layer's `lookup()` with `PoolingParams` and compares the result,
element-by-element, against the reference reduction of the same data. The matrix is the cartesian
product of the dimensions below, classified by `Classify()` in
[`layer_pooling_harness.hpp`](layer_pooling_harness.hpp) into **Enabled** (runs, must pass),
**Disabled** (a real but currently-unsupported combination — compiled and listed, skipped by
default), or dropped (degenerate / not a meaningful combination).

`Classify()` is the single source of truth; this document is its prose companion. When support for a
gap lands, flip the relevant branch in `Classify()` from `Disabled` to `Enabled` and update the
counts here.

## Dimensions

| Dimension | Values |
|---|---|
| Layer | `GPU` (cuEmbed), `LinearUVM`, `HierWithGPU`, `Host`, `HierWithoutGPU` |
| Pooling | `Concatenate` (no reduction: conversion/dequant or same-type raw passthrough), `Sum`, `Mean`, `WeightedSum`, `WeightedMean` |
| Sparse layout | `Fixed`, `CSR` |
| Input dtype | `Float32`, `Float16`, `QInt8RowwiseF32`, `QInt8RowwiseF16`, `QUint8RowwiseF32`, `QUint8RowwiseF16` |
| Output dtype | `Float32`, `Float16` |
| Weight dtype | `Float32`, `Float16` (weighted modes only; non-weighted cases use `Unknown`) |
| Key type | `int64`, `int32` |

Fixed (non-varying) parameters, by design — this suite checks **correctness**, not large-dimension
stress: `value_count = 64`, `hotness = 8`, `num_keys = 1024`, `num_rows = 65536`.

## Support matrix (current)

Generated matrix totals: **1048 enabled**, **336 disabled** (1384 generated enabled/disabled
cases), plus two standalone validation tests covering required output type and metadata-free
`Concatenate` validation.
The `HierWithoutGPU` cases require the optional PHMap plugin and are skipped when that feature is not
enabled; building `layer_pooling_test` also builds the plugin when it is available.

| Layer | Enabled | Disabled | What works | Known gaps (disabled) |
|---|--:|--:|---|---|
| **GPU** | 152 | 112 | `Float32`/`Float16` input through cuEmbed (output and weight dtype match input); matching-precision quant dequant (`Q*RowwiseF32→Float32`, `Q*RowwiseF16→Float16`) through the gathered-row CUDA path with either float weight dtype; all pooling modes, `Fixed` + `CSR`, and int64 + int32 keys/offsets | • Cross-precision quant output is not implemented by the CUDA dequant kernels. |
| **LinearUVM** | 168 | 112 | `Float32`/`Float16` (output == input, **any float weight dtype**); quant input dequantized to its scale precision (`Q*RowwiseF32→Float32`, `Q*RowwiseF16→Float16`) with **any float weight dtype**; all pooling modes; `Fixed` + `CSR`; int64 + int32 | • Cross-precision quant output (e.g. `QInt8RowwiseF32 → Float16`) — the GPU dequant/combine kernels emit only the scale's float precision. |
| **HierWithGPU** | 168 | 112 | Same dtype/pooling support as LinearUVM; the hierarchy has a GPU cache tier followed by a full `LinearHostTable` backing tier | Same cross-precision output gap as LinearUVM. |
| **Host** | 280 | 0 | All generated cases: all inputs, **any** `Float32`/`Float16` output including cross-precision, **any float weight dtype**, all pooling modes, both sparse markers, both key types | — |
| **HierWithoutGPU** | 280 | 0 | Same generated-case support as Host via the shared CPU pool/dequant path; a bounded PHMap cache is followed by a full `LinearHostTable` backing tier | — |

### Disabled groups (gtest instantiation → reason)

| Instantiation | Count | Reason |
|---|--:|---|
| `DISABLED_GPU/*` | 112 | Cross-precision quant-output cases |
| `DISABLED_LinearUVM/*` | 112 | UVM dequant kernels emit only the scale's float precision (no cross-precision output) |
| `DISABLED_HierWithGPU/*` | 112 | Same as LinearUVM |

Running the disabled groups (`--gtest_also_run_disabled_tests`) is expected to **fail**: GPU
cross-precision requests throw clear errors; the UVM/HierWithGPU cross-precision gaps produce a
value mismatch. They never abort the process.

### Excluded (degenerate) combinations — not generated

These are not capability gaps; they are not meaningful combinations, so they are dropped rather than
disabled:

- **Non-quant input with `output != input`** — there is no dequantization for float inputs, so the
  output dtype always equals the input dtype (no float→float cast dimension).
- **GPU weighted pooling with `weight_dtype != input_dtype`** — cuEmbed requires matching value and
  weight types, so these cases are dropped as unsupported.

`Concatenate` cases deliberately pass invalid sparse layout, null/negative key-index metadata, and
an irrelevant Float64 weight pointer/type. Every layer must ignore those fields and use only the
keys vector plus `output_type`; this also verifies the GPU layer's metadata-free concatenate path.

## Coverage by capability quadrant

The two orthogonal features under test are **pooling** (pooling mode ≠ `Concatenate`) and **dequant**
(quantized input dtype). The four combinations, per layer (counts are **enabled** cases that run and
verify; a parenthetical is **disabled** cases — a tracked gap that does not run):

| Layer | 1. pooling, no dequant | 2. pooling + dequant | 3. no pooling, no dequant | 4. no pooling + dequant |
|---|---|---|---|---|
| **GPU** | ✅ 32 | ✅ 96 (+96 disabled) | ✅ 8 | ✅ 16 (+16 disabled) |
| **LinearUVM** | ✅ 48 | ✅ 96 (+96 disabled) | ✅ 8 | ✅ 16 (+16 disabled) |
| **HierWithGPU** | ✅ 48 | ✅ 96 (+96 disabled) | ✅ 8 | ✅ 16 (+16 disabled) |
| **Host** | ✅ 48 | ✅ 192 | ✅ 8 | ✅ 32 |
| **HierWithoutGPU** | ✅ 48 | ✅ 192 | ✅ 8 | ✅ 32 |

- **LinearUVM / HierWithGPU / Host / HierWithoutGPU** cover all four quadrants with enabled tests.
- **GPU / LinearUVM / HierWithGPU** use the CUDA dequant kernels. Their disabled counts in quadrants
  2 and 4 are the **cross-precision** quant cases (e.g. `QInt8RowwiseF32 → Float16`); matching-
  precision dequant is enabled. The GPU layer first gathers complete raw quantized rows, then uses
  the same gathered-row CUDA path as a GPU-backed hierarchy.

## Reference & verification approach

- Each layer is backed by a linear table of `num_rows` rows indexed by key: a device buffer (GPU), a
  UVM-backed `GpuTable` (LinearUVM), a GPU-cache-plus-`LinearHostTable` hierarchy (HierWithGPU), or
  a `LinearHostTable` (Host). HierWithoutGPU uses a bounded PHMap cache in front of a
  `LinearHostTable` containing the full table. Key `k` therefore always resolves to the backing
  buffer's row `k`.
- The reference gathers rows directly from that same host buffer by key and reduces them with
  `MockHostTable<IndexT>::combine()` — the single oracle for pooling + dequant (it ignores the CUDA
  context, so the reference path is independent of the layer).
- Quant row strides are **not** padded, so `combine()`'s derived `value_count = stride - meta`
  stays exact.
- The base absolute tolerance is `1e-6` for FP32-only cases and `1e-3` when either the input or
  output is FP16-like. FP16-like `Sum` and `WeightedSum` cases multiply that base tolerance by
  `max(hotness, 1)` to account for error growth while accumulating a bag.
- The reported hitrate is asserted to be `1.0` for GPU and Host. HierWithoutGPU begins with an empty
  PHMap cache, so its initial per-tier hitrates are asserted as `0.0` for PHMap and `1.0` for the
  linear backing table. The request is repeated after the default insert heuristic warms the bounded
  cache, and the warmed per-tier hit rates must still sum to `1.0`. The LinearUVM/HierWithGPU
  pooled+dequant path does not expose the same per-key metric,
  matching the existing `UVMPooling` / `RunQuantUVMLookup` tests, which also skip that check.
- Every generated pooling request sets an explicit Float32/Float16 output type; all layers reject
  `output_type == Unknown`. A focused embedding-layer test separately covers same-type quantized
  `Concatenate`, which returns the complete raw row including scale/offset metadata.
- The current harness uses `ScopedDevice(0)` and CUDA pinned-host allocations for every layer kind,
  so even Host and HierWithoutGPU cases require a CUDA-capable test environment.

## Build & run

Built into `build_release` (configure once if needed, then):

```bash
cmake --build build_release --target layer_pooling_test -j

# All enabled tests (1048 generated matrix cases + 2 validation tests):
./build_release/bin/layer_pooling_test

# Enumerate cases (enabled), or all incl. disabled:
./build_release/bin/layer_pooling_test --gtest_list_tests
./build_release/bin/layer_pooling_test --gtest_also_run_disabled_tests --gtest_list_tests

# One case:
./build_release/bin/layer_pooling_test \
  --gtest_filter='*LinearUVM_Sum_CSR_inQI8F32_outF32_k64*'

# Per-layer shard (house-style CI sharding; matches the gpu_table_test idiom):
./build_release/bin/layer_pooling_test --gtest_filter='LinearUVM/*'
compute-sanitizer ./build_release/bin/layer_pooling_test --gtest_filter='LinearUVM/*'

# Balanced sharding across N processes (no ctest needed):
GTEST_TOTAL_SHARDS=4 GTEST_SHARD_INDEX=0 ./build_release/bin/layer_pooling_test

# Track the known gaps (expected to fail until the feature lands):
./build_release/bin/layer_pooling_test --gtest_also_run_disabled_tests --gtest_filter='DISABLED_*'
```

## Extending the matrix

1. Add a value to the relevant dimension array in `GenerateCases()`.
2. Encode its support in `Classify()` (`Enabled` / `Disabled` / drop) and, for a new gap, a line in
   `DisabledReason()`.
3. Update the counts and rows in this document.

Promoting a gap after its underlying implementation lands is a one-line change in `Classify()`.
