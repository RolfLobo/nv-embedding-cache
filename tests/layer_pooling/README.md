# Layer Pooling & Dequantization Test Suite

`layer_pooling_test` is a unified, capability-driven suite that validates embedding-layer
**pooling** and **dequantization** across **all** layer types against a single reference
(`MockHostTable::combine()`).

Each generated case drives a real layer's `lookup()` with `PoolingParams` and compares the result,
element-by-element, against the reference reduction of the same data. The matrix is the cartesian
product of the dimensions below, classified by `Classify()` in
[`layer_pooling_harness.hpp`](layer_pooling_harness.hpp) into **Enabled** (runs, must match the
reference), **Rejected** (a real but currently-unsupported combination — runs, and the layer must
refuse it with an `InvalidArgumentError`), or dropped (degenerate / not a meaningful combination).

Rejected cases are *not* skipped: a tracked gap that silently starts returning a wrong result
instead of throwing is a regression, so the refusal itself is asserted.

`Classify()` is the single source of truth; this document is its prose companion. When support for a
gap lands, flip the relevant branch in `Classify()` from `Rejected` to `Enabled` and update the
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

Generated matrix totals: **1160 enabled**, **336 rejection** (1496 generated cases), plus seven
standalone validation tests (`LayerPoolingValidationTest`) covering required output type, malformed
`PoolingParams`, metadata-free `Concatenate`, output-row layout, empty CSR bags, and padded output
strides. 1503 tests in total; none are gtest-disabled.
The `HierWithoutGPU` cases require the optional PHMap plugin and are skipped when that feature is not
enabled; building `layer_pooling_test` also builds the plugin when it is available.

| Layer | Enabled | Rejection | What works | Known gaps (rejection cases) |
|---|--:|--:|---|---|
| **GPU** | 152 | 112 | `Float32`/`Float16` input through cuEmbed (output and weight dtype match input); matching-precision quant dequant (`Q*RowwiseF32→Float32`, `Q*RowwiseF16→Float16`) through the gathered-row CUDA path with either float weight dtype; all pooling modes, `Fixed` + `CSR`, and int64 + int32 keys/offsets | • Cross-precision quant output is not implemented by the CUDA dequant kernels. |
| **LinearUVM** | 168 | 112 | `Float32`/`Float16` (output == input, **any float weight dtype**); quant input dequantized to its scale precision (`Q*RowwiseF32→Float32`, `Q*RowwiseF16→Float16`) with **any float weight dtype**; all pooling modes; `Fixed` + `CSR`; int64 + int32 | • Cross-precision quant output (e.g. `QInt8RowwiseF32 → Float16`) — the GPU dequant/combine kernels emit only the scale's float precision. |
| **HierWithGPU** | 168 | 112 | Same dtype/pooling support as LinearUVM; the hierarchy has a GPU cache tier followed by a full `LinearHostTable` backing tier | Same cross-precision output gap as LinearUVM. |
| **Host** | 336 | 0 | All generated cases: all inputs, **any** `Float32`/`Float16` output — including quant cross-precision **and** float→float casts (`Float32`→`Float16` and back), **any float weight dtype**, all pooling modes, both sparse markers, both key types | — |
| **HierWithoutGPU** | 336 | 0 | Same generated-case support as Host via the shared CPU pool/dequant path; a bounded PHMap cache is followed by a full `LinearHostTable` backing tier | — |

### Rejection groups (gtest instantiation → reason)

These run under `LayerPoolingRejectionTest.LookupRejectsUnsupportedCombo`, which asserts the layer
throws `InvalidArgumentError` instead of producing output.

| Instantiation | Count | Reason | Error |
|---|--:|---|---|
| `Rejected_GPU/*` | 112 | Cross-precision quant-output cases | `Quantized GPU embedding output type must match the quantization metadata precision` |
| `Rejected_LinearUVM/*` | 112 | UVM dequant kernels emit only the scale's float precision (no cross-precision output) | `Unsupported find_and_pool value/output type combination: ...` |
| `Rejected_HierWithGPU/*` | 112 | Same as LinearUVM | Same as LinearUVM |

### Excluded (degenerate) combinations — not generated

These are not capability gaps; they are not meaningful combinations, so they are dropped rather
than tracked as rejection cases:

- **Non-quant input with `output != input`, on the GPU-backed layers only** — the CUDA
  pool/dequant kernels have no float→float cast, so `GPU`/`LinearUVM`/`HierWithGPU` always emit the
  input dtype. The CPU pooling path dispatches its output dtype independently of the input
  (`cpu_kernel_pooling_dispatch_out`), so `Host`/`HierWithoutGPU` **do** generate these cases
  (`Float32`→`Float16` and back) and they are enabled.
- **GPU weighted pooling with `weight_dtype != input_dtype`** — cuEmbed requires matching value and
  weight types, so these cases are dropped as unsupported.

`Concatenate` cases deliberately pass invalid sparse layout, null/negative key-index metadata, and
an irrelevant Float64 weight pointer/type. Every layer must ignore those fields and use only the
keys vector plus `output_type`; this also verifies the GPU layer's metadata-free concatenate path.

## Coverage by capability quadrant

The two orthogonal features under test are **pooling** (pooling mode ≠ `Concatenate`) and **dequant**
(quantized input dtype). The four combinations, per layer (counts are **enabled** cases that run and
verify against the reference; a parenthetical is **rejection** cases — a tracked gap that runs and
must throw):

| Layer | 1. pooling, no dequant | 2. pooling + dequant | 3. no pooling, no dequant | 4. no pooling + dequant |
|---|---|---|---|---|
| **GPU** | ✅ 32 | ✅ 96 (+96 rejection) | ✅ 8 | ✅ 16 (+16 rejection) |
| **LinearUVM** | ✅ 48 | ✅ 96 (+96 rejection) | ✅ 8 | ✅ 16 (+16 rejection) |
| **HierWithGPU** | ✅ 48 | ✅ 96 (+96 rejection) | ✅ 8 | ✅ 16 (+16 rejection) |
| **Host** | ✅ 96 | ✅ 192 | ✅ 16 | ✅ 32 |
| **HierWithoutGPU** | ✅ 96 | ✅ 192 | ✅ 16 | ✅ 32 |

- **LinearUVM / HierWithGPU / Host / HierWithoutGPU** cover all four quadrants with enabled tests.
- **Host / HierWithoutGPU** quadrants 1 and 3 are twice the GPU-backed counts because the CPU path
  also casts float→float (`Float32`→`Float16` and back), which the CUDA kernels cannot.
- **GPU / LinearUVM / HierWithGPU** use the CUDA dequant kernels. Their rejection counts in quadrants
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

# Everything (1160 generated matrix cases + 336 rejection cases + 7 validation tests):
./build_release/bin/layer_pooling_test

# Enumerate cases:
./build_release/bin/layer_pooling_test --gtest_list_tests

# One case:
./build_release/bin/layer_pooling_test \
  --gtest_filter='*LinearUVM_Sum_CSR_inQI8F32_outF32_k64*'

# Per-layer shard (house-style CI sharding; matches the gpu_table_test idiom):
./build_release/bin/layer_pooling_test --gtest_filter='LinearUVM/*'
compute-sanitizer ./build_release/bin/layer_pooling_test --gtest_filter='LinearUVM/*'

# Balanced sharding across N processes (no ctest needed):
GTEST_TOTAL_SHARDS=4 GTEST_SHARD_INDEX=0 ./build_release/bin/layer_pooling_test

# Just the known gaps (each must throw; these pass until the feature lands):
./build_release/bin/layer_pooling_test --gtest_filter='Rejected_*'
```

## Extending the matrix

1. Add a value to the relevant dimension array in `GenerateCases()`.
2. Encode its support in `Classify()` (`Enabled` / `Rejected` / drop) and, for a new gap, a line in
   `RejectionReason()`.
3. Update the counts and rows in this document.

Promoting a gap after its underlying implementation lands is a one-line change in `Classify()`.
