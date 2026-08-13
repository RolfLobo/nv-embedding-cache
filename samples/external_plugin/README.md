# External Plugin Sample (pure-C ABI)

This sample demonstrates an **external-mode** NVE table plugin: a shared
object built with a potentially different toolchain than the host, which
therefore talks to NVE exclusively through the pure-C plugin interface
(`plugin/nve_external_plugin.h`). The plugin is a host-memory chained hash map
written in plain C.

```
consumer (C API)  →  Plugin loader  →  libnve-plugin-external-map.so
                        (dlopen,          pure C: opaque self pointers +
                         RTLD_LOCAL)      size-prefixed vtables of function
                                          pointers, nve_status_t returns
```

## What this sample shows

1. **The external plugin contract** — the plugin includes the single pure-C
   header (`plugin/nve_external_plugin.h`) and **does not link `nve-common`**.
   It emits the five metadata symbols and the idempotent
   NVE-services handshake with one macro:
   `NVE_DEFINE_EXTERNAL_PLUGIN("Sample External Map", "NVIDIA Corporation")`,
   and implements the well-known creation symbol
   `nve_plugin_create_external_factory`.

2. **Output ownership** — factory creation and `produce` zero their output
   first and transfer a nonzero handle only on `NVE_SUCCESS`; every error path
   releases the plugin's own partial allocations.

3. **NVE services** — the plugin stores the size-prefixed
   `nve_ext_nve_services_t` table it receives in the handshake, checks each
   service with `NVE_ABI_HAS_FIELD` before use, and parallelizes large `find`
   batches through the status-returning `execute_n` threadpool service
   (64-key-aligned chunks keep each hit-mask word private to one task).

4. **Error reporting** — fallible ops return `nve_status_t` and keep a
   thread-local message retrievable via the `last_error` callback; no
   exception ever crosses the C boundary.

5. **Consuming the plugin** — the consumer selects the plugin by shared-object
   name in one call:

   ```c
   nve_table_factory_t factory = NULL;
   CHECK_NVE(nve_create_table_factory(&factory, "libnve-plugin-external-map.so",
                                      "{\"num_buckets\": 512}"));
   ```

   It then exercises every table operation (`find` including a count-only
   `NULL`-values probe, `insert`, `update`, `update_accumulate`, `erase`,
   `clear`, and both lookup-counter calls) and finally drives the same table
   behind a hierarchical embedding layer, exactly like any in-tree table.

## Table configuration

JSON passed to `nve_table_factory_produce` (all optional):

| Key              | Default | Meaning                                   |
|------------------|---------|-------------------------------------------|
| `max_value_size` | 256     | Fixed row size in bytes                   |
| `num_buckets`    | factory default (1024, or factory JSON `num_buckets`) | Hash bucket count |
| `invalid_key`    | -1      | Sentinel key value reported to the host   |

The table stores 8-byte keys and reports `NVE_DTYPE_FLOAT32` values.

Concurrency: concurrent `find` calls are safe; modify operations must not run
concurrently with any other call on the same table (see the contract comment
in `src/plugin.c`).

## Directory layout

```
samples/external_plugin/
├── CMakeLists.txt              # Builds plugin (C only) + consumer
├── README.md
├── external_plugin_sample.cpp  # Consumer executable (NVE C API only)
└── src/
    └── plugin.c                # The pure-C plugin
```

## Building and running

Built automatically as part of the default CMake build:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)

./bin/external_plugin_sample
# external plugin sample: OK
```

The consumer loads the plugin by bare soname, which resolves through the
loader's fallback next to `libnve-common.so`; an absolute path works the same
way.
