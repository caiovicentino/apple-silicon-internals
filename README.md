# apple-silicon-internals

Reverse engineering toolkit for Apple Silicon. Discovers undocumented private APIs, maps the full SoC telemetry surface, and provides Claude Code skills for hardware introspection.

Built on macOS 26 Tahoe (developer beta), Apple M4 Mac mini. Based on the runtime introspection approach from [maderix/ANE](https://github.com/maderix/ANE).

> **Note**: macOS 26 is unreleased. Some APIs documented here (especially Metal 4 / MTL4 classes) may be new to this version and not present on macOS 15 Sequoia.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Skills (Claude Code)](#skills-claude-code)
- [Discoveries](#discoveries)
  - [Apple Intelligence On-Device Models](#apple-intelligence-on-device-models)
  - [Metal 4 Machine Learning Pipeline](#metal-4-machine-learning-pipeline)
  - [GPU Tensor API](#gpu-tensor-api)
  - [SoC Telemetry (1009 Channels)](#soc-telemetry-1009-channels)
  - [CLPC Performance Controller](#clpc-performance-controller)
  - [Apple M4 Architecture](#apple-m4-architecture)
  - [Vector Database & Embeddings](#vector-database--embeddings)
  - [Hidden ML Runtimes](#hidden-ml-runtimes)
  - [100+ Apple Intelligence Use Cases](#apple-intelligence-use-cases)
  - [macOS 26 New APIs (~455 classes)](#macos-26-new-apis)
  - [M5 GPU Firmware](#m5-gpu-firmware)
- [Benchmarks](#benchmarks)
- [Repository Structure](#repository-structure)
- [Building](#building)
- [Methodology](#methodology)
- [Limitations](#limitations)
- [Legal](#legal)

---

## Quick Start

```bash
git clone <repo> && cd apple-silicon-internals
make all

# Map the SoC
tools/full_soc_map

# Monitor power in real-time (no sudo!)
probes/soc_power 5 1000

# Scan any private framework
tools/framework_scanner AppleNeuralEngine

# X-ray an installed app
tools/framework_scanner PhotosIntelligence | grep CLASS

# Benchmark the hardware
benchmarks/inference_engine 512
```

---

## Skills (Claude Code)

10 slash commands that give Claude direct access to Apple Silicon hardware. Install by cloning this repo as your working directory in Claude Code.

| Skill | Description |
|-------|-------------|
| `/silicon-power` | Real-time power consumption per SoC subsystem (no sudo) |
| `/silicon-profile <cmd>` | Measure the energy cost of any command |
| `/silicon-watch` | Continuous SoC monitoring |
| `/silicon-soc` | Complete hardware spec sheet |
| `/silicon-bench` | CPU/GPU benchmarks and model capacity analysis |
| `/silicon-scan <framework>` | Scan private frameworks for undocumented APIs |
| `/silicon-xray <app>` | X-ray any app's private framework usage and entitlements |
| `/silicon-entitlements <binary>` | Dump entitlements of any binary or app |
| `/silicon-ocr <image>` | On-device OCR via Apple Vision |
| `/silicon-detect <text>` | On-device language detection |
| `/silicon-deepxray <app>` | Complete 11-phase app reverse engineering |
| `/silicon-audit <app>` | Privacy/security audit with risk assessment |
| `/silicon-compare <app1> <app2>` | Side-by-side app comparison (entitlements, frameworks, tracking) |
| `/silicon-shaders [filter]` | Dump and analyze all 265 Metal shader libraries |
| `/silicon-tts <text>` | Text-to-speech audio generation (184 voices, pt-BR) |
| `/silicon-ner <text>` | Named entity recognition + data type detection |

16 skills total. No MCP server needed — they work through Claude Code's native skill system.

---

## Discoveries

### Apple Intelligence On-Device Models

We mapped Apple's complete on-device model inventory by scanning the MobileAsset system and GenerativeModels framework.

**Architecture**: One 3B base model + task-specific LoRA adapters. One 300M model for safety/utility.

#### 3B Model — 28 LoRA Adapters

| Adapter | Task |
|---------|------|
| `base` | Foundation model |
| `summarization` | Text summarization |
| `magic_rewrite` | Text rewriting |
| `concise_tone` / `friendly_tone` / `professional_tone` | Tone adjustment |
| `mail_reply` | Email reply generation |
| `messages_reply` / `messages_action` | iMessage responses |
| `proofreading_review` | Grammar/spelling review |
| `photos_memories_title` | Photo memory titles |
| `photos_memories_asset_curation_outlier` | Photo curation |
| `shortcuts_ask_afm_action_3b` / `_v2` | Shortcuts actions |
| `auto_tagger` / `fm_api_content_tagger` | Content tagging |
| `text_event_extraction` / `text_person_extraction` | Named entity recognition |
| `urgency_classification` | Priority detection |
| `autonaming_messages` | Conversation naming |
| `suggest_recipe_items` | Recipe suggestions |
| `fm_api_generic` | General purpose API |

#### 300M Model — Safety & Utilities

| Model | Task |
|-------|------|
| `safety` / `misc_safety` / `prepubescent_safety` | Content safety filters |
| `factual_consistency_classifier` | Hallucination detection |
| `vi_content_classifier` | Visual content classification |
| `image_tokenizer` | Image tokenization |
| `structural_integrity` | Output structure validation |
| `adm_people_grounding` / `adm_prompt_rewriting` | Prompt preprocessing |

#### Server Models (Private Cloud Compute)

28 server-side adapters for tasks requiring more compute: long-form mail, multimodal understanding, complex rewriting. These run on Apple's PCC infrastructure, not on-device.

#### Execution Chain

```
App (com.apple.modelmanager.inference)
  → GenerativeModelsFoundation.ModelCache
    → MobileAsset (com.apple.MobileAsset.UAF.FM.GenerativeModels)
  → siriinferenced / generativeexperiencesd
    → AjaxLLM (session + model)
      → aned (com.apple.ane.iokit-user-access)
        → ANE hardware
```

Key entitlements required:
- `com.apple.modelcatalog.ajax` — access to the Ajax model catalog
- `com.apple.modelmanager.inference` — permission to run inference
- `com.apple.MobileAsset.UAF.FM.GenerativeModels` — access to model assets
- `com.apple.private.security.storage.MobileAssetGenerativeModels` — read model files

These entitlements require Apple-signed provisioning profiles and cannot be self-signed.

#### Photos.app — 282 Hidden AI Classes

Photos has its own intelligence framework (`PhotosIntelligence`) containing:
- `AjaxLLM` — local LLM for generating titles, stories, and captions
- `PromptSuggestionSafetyValidator` — validates prompts before sending to LLM
- `PersonalTraitGenerator` — builds personality profiles from photo patterns
- `FreeformStoryGenerator` — generates narratives from photo collections
- `StoryMusicCurator` — selects music for memories using ML keywords

#### Running Daemons

| Process | Function |
|---------|----------|
| `siriinferenced` | Main inference daemon (Mach service) |
| `intelligenceplatformd` | Platform orchestration |
| `IntelligencePlatformComputeService` | Compute dispatch (XPC) |
| `HostInferenceProviderService` | Model execution (runs as `_modelmanagerd`) |
| `intelligencecontextd` | Context flow |
| `siriknowledged` | Knowledge base |
| `generativeexperiencesd` | Generative experiences orchestration |

---

### Metal 4 Machine Learning Pipeline

93 MTL4 classes found in the macOS 26 runtime. Apple is building a dedicated ML execution path in Metal 4.

```
MPSGraph (MLIR) → compile → MPSGraphExecutable
    → _MTL4MachineLearningPipelineState (wraps MPSGraphExecutableProxy)
    → MTL4MachineLearningCommandEncoder.dispatchNetworkWithIntermediatesHeap:
    → AGXG16GFamilyCommandQueue_mtlnext
```

Key classes:
- `MTL4MachineLearningPipelineDescriptor` — describes an ML pipeline
- `_MTL4MachineLearningPipelineState` — compiled pipeline (wraps `MPSGraphExecutableProxy`)
- `IOGPUMetal4MachineLearningCommandEncoder` — encodes ML dispatch to GPU
- `MTL4FunctionDescriptor` / `MTL4LibraryFunctionDescriptor` — function descriptors

The pre-compiled `MPSGraphExecutable` path vs standard `MPSGraph.run()`:

| Model dimensions | Standard | Pre-compiled | Speedup |
|------------------|----------|-------------|---------|
| GPT-2 124M | 30.7 ms | 11.3 ms | **2.72x** |
| Llama-1B | 54.7 ms | 48.3 ms | **1.13x** |
| Qwen-3B | 202 ms | 187 ms | **1.08x** |
| Qwen-4B | 168 ms | 152 ms | **1.10x** |

*Single transformer layer, batch=1, seq=128, FP16. Speedup comes from eliminating graph interpretation overhead — larger models are compute-bound so the speedup decreases.*

---

### GPU Tensor API

Metal has a native tensor type not in any public documentation:

```objc
[device supportsTensors];                        // YES on M4/macOS 26
[device newTensorWithDescriptor:desc error:&err]; // → AGXG16GFamilyTensor
```

- `MTLTensorDescriptor` — dimensions, strides, dataType, usage, storageMode
- `MTLTensorExtents` — `initWithRank:values:` for dimension/stride specs
- Tensor operations: `newTensorViewWithReshapedDescriptor:`, `newTensorViewWithSlice:`, `replaceSlice:withBytes:strides:`

We successfully created a `[4, 256, 256]` float32 tensor (1 MB) on the M4 GPU.

llama.cpp already knows this API exists but disables it: `tensor API disabled for pre-M5 and pre-A19 devices`. This suggests Apple will officially expose it with the next chip generation.

---

### SoC Telemetry (1009 Channels)

Via `libIOReport.dylib`, loaded with `dlsym()` at runtime. **No sudo required on macOS 26.**

| Group | Channels | What it measures |
|-------|----------|-----------------|
| AMC Stats | 133 | Per-subsystem memory bandwidth (ECPU, PCPU, GPU, ANE, ISP, AVE, Display, SEP, PCIe), DRAM CAS/RAS counts, self-refresh cycles |
| Energy Model | 175 | Per-core energy in mJ (ECPU0-5, PCPU0-3), per-core SRAM, GPU, GPU SRAM, ANE, ISP, AVE, DRAM, DCS, Display, PCIe, SOC_AON |
| GPU Stats | 181 | Per-zone thermals (Tg0a-Tg14a), power zone filters, throttle counters, idle cycles |
| CPU Stats | 16 | Per-core P-state residency (IDLE, V0P7 through V7P0), per-cluster voltage states |
| PMP | 494 | Power management counters |
| NVMe | 10 | Storage performance |

For comparison: `asitop` (4.5K GitHub stars) requires sudo and shows ~20 data points. We expose 1009 channels without sudo.

---

### CLPC Performance Controller

`PerformanceControlKit.framework` provides direct SoC control:

- `CLPCPolicyClient.setGameMode:options:error:` — toggle Game Mode
- `CLPCPolicyClient.isLowPowerModeCandidate:error:` — query LPM eligibility
- `CLPCReportingClient` — 1208 bytes, 30 ivars including `num_cpu_clusters`, `num_ane_clusters`, `num_cpu_cores`, 11 schema types

Requires `com.apple.private.clpc` entitlement for full access.

---

### Apple M4 Architecture

| Component | Details |
|-----------|---------|
| **SoC** | Apple M4 (t8132), Mac mini Mac16,10 (J773gAP) |
| **E-cores** | 6x `apple,sawtooth`, L1d=64KB, L1i=128KB, L2=4MB shared |
| **P-cores** | 4x `apple,everest`, L1d=128KB, L1i=192KB, L2=16MB shared |
| **GPU** | `AGXG16GDevice`, MTLGPUFamilyApple9, Metal 4 (5002) |
| **Memory** | 16 GB unified, 128-byte cache line, 12.1 GB GPU working set |
| **SME** | SME2 (512-bit tiles), BF16, I8MM, F32F32, I8I32, F16F32 |
| **Features** | Raytracing, mesh shaders, function pointers, BF16, dynamic libraries |

---

## Benchmarks

### Real Model: Qwen3.5-4B (Q8_0, 4.2 GB GGUF)

Running via llama.cpp on the same M4:

```
Prompt eval:  37.85 tokens/sec (528 ms for 20 tokens)
Generation:   17.18 tokens/sec (58 ms per token)
GPU family:   MTLGPUFamilyMetal4 (5002)
Backend:      BLAS + Metal (all layers on GPU)
```

### Compute Throughput

| Backend | Measured | Notes |
|---------|----------|-------|
| CPU FP32 (Accelerate/SME2) | 1.65 TFLOPS | `cblas_sgemm` 2048x2048 |
| GPU FP16 (Metal compute) | 5.41 TFLOPS | Pre-compiled MPSGraphExecutable |
| GPU FP16 (naive shader) | 342 GFLOPS | Unoptimized matmul kernel |
| ANE IOSurface I/O | 1479 GB/s | Zero-copy round-trip |

### Memory Capacity

| Precision | Max inference | Max full training | Max LoRA/QLoRA |
|-----------|--------------|-------------------|----------------|
| FP16 | ~7B params | ~800M params | — |
| INT8 | ~13B params | — | ~11B base |
| INT4 | ~26B params | — | ~19B base |

### What We Tested That Didn't Help

| Experiment | Result |
|-----------|--------|
| `GGML_METAL_TENSOR_ENABLE=1` | No improvement (pp -5%, tg +1%) |
| MPSGraph vs llama.cpp shaders | llama.cpp 3x faster (custom SIMD kernels) |
| MPS quantized ops | Slower than llama.cpp's fused shaders |
| BNNS MultiheadAttention | Attention is 0.003ms — not the bottleneck |
| Various thread counts | tg identical (GPU-bound) |

**Conclusion**: Token generation at 19 tok/s is memory-bandwidth-bound (~100 GB/s DRAM). No software API change helps — you need either more bandwidth (M4 Pro/Max) or better quantization.

---

## Repository Structure

```
.claude/skills/                 # 10 Claude Code skills
  silicon-power/                #   Real-time power monitoring
  silicon-profile/              #   Energy profiling of commands
  silicon-watch/                #   Continuous SoC monitoring
  silicon-soc/                  #   Hardware spec sheet
  silicon-bench/                #   Compute benchmarks
  silicon-scan/                 #   Framework scanning
  silicon-xray/                 #   App X-ray (frameworks + entitlements)
  silicon-entitlements/         #   Entitlement dumping
  silicon-ocr/                  #   On-device OCR
  silicon-detect/               #   Language detection

tools/                          # Discovery tools
  framework_scanner.m           #   Scan any framework for classes/methods
  deep_probe.m                  #   Deep introspection with instantiation
  batch_scan.m                  #   Batch scan by category
  full_soc_map.m                #   Map all 1009 IOReport channels

probes/                         # Hardware probes
  gpu_probe.m                   #   GPU internals, Metal 4, tensors
  clpc_probe.m                  #   CLPC performance controller
  intelligence_probe.m          #   Apple Intelligence frameworks
  soc_power.m                   #   Real-time power via IOReport

pocs/                           # Proof of concepts
  metal_tensor_poc.m            #   Create GPU tensors (works!)
  mtl4ml_engine.m               #   MTL4 ML pipeline benchmark
  ane_direct.m                  #   ANE direct access (MIL rejected)
  gamemode_ctl.m                #   Game Mode control (needs entitlement)

benchmarks/                     # Performance tests
  real_model_bench.m            #   Transformer layers at real model dims
  inference_engine.m            #   CPU vs GPU vs ANE comparison
  capacity_test.m               #   Memory limits and model calculator
  advantage_test.m              #   Private vs public API overhead
  mps_quant_bench.m             #   MPS quantized ops benchmark
  cpu_attn_bench.m              #   CPU attention bottleneck analysis
  bnns_mha_bench.m              #   BNNS MultiheadAttention test
  kvcache_compress.m            #   KV cache compression (TurboQuant)

docs/
  FINDINGS.md                   #   Detailed technical findings
  ROADMAP.md                    #   What to build next
```

---

## Building

```bash
# Requires: macOS 26, Apple Silicon, Xcode Command Line Tools
xcode-select --install

# Build everything
make all

# Build specific targets
make tools        # Discovery tools
make probes       # Hardware probes
make pocs         # Proof of concepts
make benchmarks   # Performance tests

# Quick targets
make soc-map      # Map all IOReport channels
make power        # Monitor power (5 samples)
make bench        # Run model benchmarks
```

---

## Methodology

All discovery via public Objective-C runtime introspection functions:

1. `dlopen()` loads private frameworks from `/System/Library/PrivateFrameworks/`
2. `objc_copyClassList()` + `class_getImageName()` filters classes by framework
3. `class_copyMethodList()` / `class_copyPropertyList()` / `class_copyIvarList()` dumps the full API surface
4. `objc_msgSend()` dynamically invokes discovered methods
5. `dlsym()` resolves C functions in private libraries (`libIOReport.dylib`)
6. `codesign -d --entitlements` extracts entitlements from signed binaries

Statistics: 2,152 private frameworks enumerated. 1000+ ML/AI classes discovered. 93 Metal 4 classes found. 55+ Apple Intelligence models mapped. 100+ AI use cases documented. ~455 new macOS 26 classes found. 614 CoreML models on system. 265 Metal shader libraries. 16 Claude Code skills created. See `docs/` for detailed findings:

- `DEEP_REVERSING.md` — CoreML models, ANE format, Metal shaders, XPC daemons
- `MACOS26_NEW_APIS.md` — 455 new classes (Image Playground, OSIntelligence, IntelligenceFlow)
- `VECTOR_SEARCH.md` — Native vector database + embeddings + on-device RAG
- `HIDDEN_RUNTIMES.md` — Morpheus (MLX+Python), XOJIT (JIT/LLVM), ODIE (ML VM), TuriCore
- `APPLE_INTELLIGENCE_USE_CASES.md` — 100+ use cases with safety filters
- `BLOG_POST.md` — Ready-to-publish blog post

---

## Limitations

- **Private APIs break.** Everything here may stop working on the next macOS update.
- **macOS 26 beta.** MTL4 classes, MTLTensor, and some IOReport behavior may be specific to this version.
- **No actual model inference.** Benchmarks use the same matrix dimensions as real models but don't load actual weights.
- **Entitlements.** Apple Intelligence models, Game Mode control, and ANE compilation require Apple-signed entitlements.
- **Swift classes.** Many newer frameworks use Swift — ObjC runtime introspection can see class names and ivars but not Swift method signatures.
- **Performance.** We confirmed that no discovered private API provides meaningful inference speedup over llama.cpp's optimized Metal shaders. The 10% MPSGraphExecutable improvement is a public API.

---

## Test Environment

- **Hardware**: Mac mini (Mac16,10), Apple M4, 10 cores (4P+6E), 16 GB
- **OS**: macOS 26.3.1 Tahoe (Build 25D771280a)
- **Model tested**: Qwen3.5-4B Q8_0 via llama.cpp (17.18 tok/s)

---

## Legal

Runtime introspection for interoperability research under fair use (*Sega v. Accolade*, 1992; DMCA 1201(f)). No Apple proprietary code is included. All APIs discovered via public ObjC runtime functions and `codesign` entitlement extraction.

## License

MIT
