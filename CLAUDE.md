# Apple Silicon Internals

Reverse engineering toolkit for Apple Silicon private APIs on macOS 26 (Tahoe).

## Build

```bash
make all
```

## Skills

| Command | Description |
|---------|-------------|
| `/silicon-power` | Real-time power from every SoC subsystem (no sudo) |
| `/silicon-profile <command>` | Profile the energy cost of running a command |
| `/silicon-watch [samples] [interval_ms]` | Live SoC monitoring |
| `/silicon-soc` | Complete hardware map of the SoC |
| `/silicon-bench [size]` | Benchmark CPU/GPU compute and model capacity |
| `/silicon-scan <framework>` | Scan private frameworks for undocumented APIs |
| `/silicon-xray <app>` | X-ray any app's private frameworks and entitlements |
| `/silicon-entitlements <binary>` | Dump entitlements of any binary |
| `/silicon-ocr <image>` | On-device OCR via Apple Vision |
| `/silicon-detect <text>` | On-device language detection |

## Key files

- `tools/` — Framework scanning and IOReport channel discovery
- `probes/` — GPU, CLPC, Apple Intelligence, power probes
- `pocs/` — Metal tensor, ANE direct access, MTL4 ML pipeline PoCs
- `benchmarks/` — Real model dimension benchmarks, capacity analysis

## Notes

- IOReport works without sudo on macOS 26. May require sudo on older versions.
- Private APIs may break between macOS versions.
- Metal tensor API exists on M4 but llama.cpp disables it (gates behind M5/A19).
- MTL4 classes (93 found) may be new to macOS 26.
- Apple Intelligence models (3B + 28 LoRA adapters) are on-device but require Apple-signed entitlements.
