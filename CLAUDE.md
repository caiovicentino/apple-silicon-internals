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
| `/silicon-xray <app>` | Quick X-ray of any app's private frameworks and entitlements |
| `/silicon-deepxray <app>` | Complete 11-phase reverse engineering of any app |
| `/silicon-audit <app>` | Privacy/security audit with risk assessment |
| `/silicon-compare <app1> <app2>` | Side-by-side comparison of two apps |
| `/silicon-entitlements <binary>` | Dump entitlements of any binary |
| `/silicon-shaders [filter]` | Dump and analyze all 265 Metal shader libraries |
| `/silicon-ocr <image>` | On-device OCR via Apple Vision |
| `/silicon-detect <text>` | On-device language detection |
| `/silicon-tts <text>` | Text-to-speech audio generation (184 voices, pt-BR) |
| `/silicon-ner <text>` | Named entity recognition + data detection |

## Key files

- `tools/` — Framework scanning and IOReport channel discovery
- `probes/` — GPU, CLPC, Apple Intelligence, power probes
- `pocs/` — Metal tensor, ANE direct access, MTL4 ML pipeline PoCs
- `benchmarks/` — Real model benchmarks, capacity analysis, MPS comparison
- `docs/` — Research findings (10 documents)
- `scripts/` — Helper scripts (TTS)

## Key discoveries

- 55+ Apple Intelligence on-device models (3B base + 28 LoRA adapters)
- 100+ Apple Intelligence use cases mapped from UAF configuration
- ~455 new classes in macOS 26 (Image Playground, OSIntelligence, IntelligenceFlow)
- Native vector database + embedding system (VectorSearch, SpotlightEmbedding)
- Hidden ML runtimes: Morpheus (MLX+Python), XOJIT (JIT/LLVM), ODIE (ML VM)
- Metal 4 ML pipeline (93 MTL4 classes)
- 1009 IOReport telemetry channels (no sudo on macOS 26)
- M5 GPU firmware (G17G/P/X) already in macOS 26
- 614 CoreML models on system, 265 Metal shader libraries

## Notes

- IOReport works without sudo on macOS 26. May require sudo on older versions.
- Private APIs may break between macOS versions.
- Metal tensor API exists on M4 but llama.cpp disables it (gates behind M5/A19).
- MTL4 classes (93 found) may be new to macOS 26.
- Apple Intelligence models require Apple-signed entitlements to access.
- System prompts are NOT stored as text — they're in adapter weights or protected MobileAsset downloads.
