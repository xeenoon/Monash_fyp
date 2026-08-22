# Reusable C utilities

Small cross-module operations live in focused `*_utils.h/.c` pairs and use the
module name as their public symbol prefix. Keep policy specific to a subsystem
in that subsystem; for example, terrain-header validation remains in
`terrain_tile.c`.

Current modules are:

- `byte_utils`: unaligned, host-independent little-endian integer and IEEE-754
  decoding;
- `str_utils`: allocation-based string operations with explicit ownership;
- `size_utils`: overflow-checked `size_t` arithmetic;
- `file_utils`: whole-file reads and typed error results;
- `checksum_utils`: checksum APIs backed by the system zlib implementation.

Use a standard/system implementation when it is portable and matches the
required semantics. The project targets C11, which does not provide endian
decoding, bounded string duplication, or checked integer arithmetic, so those
small adapters remain local. CRC-32 is non-trivial and already implemented by
zlib, so it is not maintained here.

Utility APIs follow these conventions:

- names start with the module prefix (`byte_`, `str_`, `size_`, etc.);
- allocated results document ownership and are released with `free`;
- fallible operations return a boolean or a typed result enum;
- output parameters are reset on failure when practical;
- utilities do not log, abort, or embed terrain/renderer policy;
- every public utility receives direct unit coverage in `tests/utils_tests.c`.
