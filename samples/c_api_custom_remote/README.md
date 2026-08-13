# Custom Remote Host Table Sample (C API)

This sample demonstrates how to implement a custom "remote" host table as an
NVE plugin and use it with the C API in a three-tier hierarchical embedding
layer:

```
GPU table  (L1, GPU SRAM / HBM cache)
  → phmap host table  (L2, CPU RAM cache)
    → custom_remote   (L3, your parameter server / DB)
```

## What this sample shows

1. **Writing a custom host table plugin** — subclass `nve::HostTableLike`,
   implement the six CRUD operations (`find`, `insert`, `update`,
   `update_accumulate`, `clear`, `erase`), and emit the plugin metadata
   symbols with the `NVE_DEFINE_INTERNAL_PLUGIN_NO_CONFIG` macro. The macro
   defines the five `nve_plugin_*` entry points (`nve_plugin_abi_version_major`,
   `nve_plugin_abi_version_minor`, `nve_plugin_name`, `nve_plugin_author`,
   `nve_plugin_mode`) plus the factory hook
   `nve_plugin_create_internal_factory`.

2. **Loading the plugin at runtime** — the sample executable selects the
   factory by shared-object path:
   `nve_create_table_factory(&factory, "libnve-plugin-custom_remote.so", "{}")`
   loads the plugin, creates its factory (the JSON carries plugin-specific
   factory options; `"{}"` when none are needed), and the factory then
   produces tables.

3. **Pure C API usage** — all layer, table, and context operations use
   `nve_c_api.h`; no C++ NVE headers are included in the sample executable.

## Directory layout

```
samples/c_api_custom_remote/
├── CMakeLists.txt                  # Builds the sample executable
├── c_api_custom_remote_sample.cpp  # Sample executable (C API only)
└── README.md
```

The plugin source is shared from `samples/common/custom_remote_plugin/`:

```
samples/common/custom_remote_plugin/
├── CMakeLists.txt                  # Builds libnve-plugin-custom_remote.so
├── include/
│   └── custom_remote_table.hpp    # CustomRemoteTable & CustomRemoteTableFactory
└── src/
    ├── custom_remote_table.cpp    # Table implementation (std::map-backed)
    └── plugin.cpp                 # Plugin entry points
```

## Building

The sample is built automatically as part of the default CMake build:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

The build produces:
- `libnve-plugin-custom_remote.so` — the plugin shared library
- `c_api_custom_remote_sample` — the sample executable

## Running

```bash
./build/bin/c_api_custom_remote_sample
```

Expected output shows:
- Creation of the three table tiers (GPU, phmap, custom_remote)
- Insertion of 1000 synthetic embeddings into the remote tier
- Lookup of 16 keys traversing all three tiers (100% remote hit rate)
- Verification that retrieved embeddings match expected values

## Adapting to a real remote table

To connect to a real parameter server, replace the `std::map` in
`CustomRemoteTable` with your network client. The key methods to modify are:

- `find()` — issue a batch GET to your KV store
- `insert()` — issue a batch PUT
- `update()` / `update_accumulate()` — issue batch UPDATE / INCREMENT
- `erase()` / `clear()` — issue batch DELETE

Factory JSON passed to `nve_create_table_factory` can carry connection strings,
timeouts, or other backend options. Table JSON passed to
`nve_table_factory_produce` describes an individual table.
