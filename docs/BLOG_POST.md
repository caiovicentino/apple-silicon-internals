# I Reverse Engineered Apple Intelligence on macOS 26

*What's inside Apple Silicon, how Apple Intelligence runs on-device, and what nobody has documented before.*

---

Last week I pointed the Objective-C runtime introspection at every private framework on my M4 Mac mini running macOS 26 Tahoe. Here's what I found.

## The Setup

macOS ships with **2,152 private frameworks** in `/System/Library/PrivateFrameworks/`. Apple uses these internally but doesn't document them. Using `dlopen()`, `objc_copyClassList()`, and `objc_msgSend()`, you can load any of them and enumerate every class, method, property, and ivar — no jailbreak needed.

I built a [toolkit](https://github.com/caiovicentino/apple-silicon-internals) that automates this, then pointed it at everything ML-related on the system.

## Finding 1: Apple Intelligence Runs a 3B Model with 28 LoRA Adapters

By scanning the MobileAsset system, I found Apple's complete on-device model inventory. It's not one monolithic model — it's a **3B base model** with task-specific LoRA adapters that get swapped in depending on what you're doing:

| Adapter | What it does |
|---------|-------------|
| `summarization` | Summarize text |
| `magic_rewrite` | Rewrite text in different style |
| `concise_tone` / `friendly_tone` / `professional_tone` | Adjust tone |
| `mail_reply` | Draft email replies |
| `messages_reply` | Reply to iMessages |
| `proofreading_review` | Grammar correction |
| `photos_memories_title` | Generate titles for photo memories |
| `auto_tagger` | Tag content |
| `text_event_extraction` / `text_person_extraction` | Named entity recognition |
| `urgency_classification` | Detect urgency |

Plus a smaller **300M model** for safety: `safety`, `prepubescent_safety`, `factual_consistency_classifier` (hallucination detection), and `structural_integrity` (output validation).

There are also 28 server-side adapters for Private Cloud Compute — heavier tasks like long-form email and multimodal understanding get routed to Apple's secure cloud.

## Finding 2: Photos Has a Hidden LLM Called "AjaxLLM"

Scanning `PhotosIntelligence.framework` revealed **282 hidden AI classes**, including:

- **`AjaxLLM`** — a local LLM class. "Ajax" is Apple's internal codename for their language models.
- **`PromptSuggestionSafetyValidator`** — validates prompts before passing to the LLM
- **`PersonalTraitGenerator`** — builds personality profiles from your photos
- **`FreeformStoryGenerator`** — generates photo story narratives
- **`StoryMusicCurator`** — picks music for memories using ML

The entitlement chain to access AjaxLLM requires:
```
com.apple.modelcatalog.ajax         → model catalog access
com.apple.modelmanager.inference     → inference permission
com.apple.MobileAsset.UAF.FM.GenerativeModels → model file access
```
These require Apple-signed provisioning profiles. You can't self-sign them.

## Finding 3: 1,009 IOReport Telemetry Channels (No Sudo)

This was a surprise: `libIOReport.dylib` exposes real-time telemetry for every subsystem on the SoC — and on macOS 26, it works **without sudo**.

- Per-core energy (ECPU0-5, PCPU0-3) in millijoules
- GPU thermals per zone (10 thermal sensors)
- DRAM bandwidth per subsystem (CPU, GPU, ANE, ISP, Display, PCIe)
- CPU P-state residency (time at each frequency level)
- GPU throttle counters

For comparison: `asitop` (4.5K GitHub stars) requires sudo and shows ~20 data points. We get 1,009.

## Finding 4: Metal 4 Has a Hidden ML Pipeline

93 MTL4 classes exist in the macOS 26 runtime. The most interesting is `MTL4MachineLearningCommandEncoder` — a dedicated GPU command encoder for ML operations.

The execution chain:
```
MPSGraph (MLIR) → compile → MPSGraphExecutable
  → _MTL4MachineLearningPipelineState
  → MTL4MachineLearningCommandEncoder.dispatchNetworkWithIntermediatesHeap:
  → AGXG16GFamilyCommandQueue_mtlnext
```

The pre-compiled `MPSGraphExecutable` path is **2.7x faster** than `MPSGraph.run()` for small models (GPT-2 scale) and ~10% faster for larger ones (4B scale). The speedup comes from eliminating graph interpretation overhead.

This is the same path Apple Intelligence uses internally. The `_MTL4MachineLearningPipelineState` wraps an `MPSGraphExecutableProxy` — confirming that MTL4 ML is built on top of MPSGraph.

## Finding 5: The GPU Tensor API Exists but Is Gated

Metal has undocumented tensor support:

```objc
[device supportsTensors];                         // YES on M4
[device newTensorWithDescriptor:desc error:&err];  // → AGXG16GFamilyTensor
```

I successfully created GPU tensors with reshape, slice, and view operations. But llama.cpp already knows about this API and **deliberately disables it**: `tensor API disabled for pre-M5 and pre-A19 devices`.

I tested with `GGML_METAL_TENSOR_ENABLE=1`: no performance gain on M4. Apple is likely still optimizing the tensor kernel paths for current hardware.

## Finding 6: 614 CoreML Models Ship with macOS

By scanning the entire filesystem:

- **68 MB** Siri TTS decoder (text-to-speech)
- **64 MB** CLIP-like text encoder (6-bit quantized, 512-token context)
- **11 MB** Language-aligned audio encoder (spectrogram → 512-dim embedding)
- **9 MB** BERT for Siri input representations
- 265 Metal shader libraries (28 MB Espresso ML kernels)
- CPU and ANE variants of the same model compiled side-by-side

## Finding 7: The M4 Architecture

| Detail | Value |
|--------|-------|
| E-cores | 6x `apple,sawtooth` |
| P-cores | 4x `apple,everest` |
| Platform | t8132 |
| GPU | AGXG16GDevice → AGXG16GFamilyDevice → IOGPUMetalDevice |
| SME | SME2 with 512-bit tiles, BF16, I8MM |
| Peak measured | 1.65 TFLOPS FP32 (CPU), 5.4 TFLOPS FP16 (GPU) |

## What Didn't Work

Being honest about what the private APIs **don't** do:

- **No inference speedup over llama.cpp.** Token generation is memory-bandwidth-bound at ~19 tok/s. No API change helps.
- **Can't access AjaxLLM.** Entitlements are Apple-signed only.
- **ANE direct MIL compilation failed.** The MIL format requires CoreML-generated programs.
- **MPS quantized ops are slower than llama.cpp's custom Metal shaders** (3x slower per matmul).

## The Toolkit

Everything is open source: **[github.com/caiovicentino/apple-silicon-internals](https://github.com/caiovicentino/apple-silicon-internals)**

Includes 10 Claude Code skills for AI agent hardware introspection:

| Skill | What it does |
|-------|-------------|
| `/silicon-power` | Real-time SoC power (no sudo) |
| `/silicon-xray <app>` | X-ray any app's private APIs |
| `/silicon-scan <framework>` | Scan any private framework |
| `/silicon-profile <cmd>` | Energy cost of a command |
| `/silicon-bench` | CPU/GPU benchmarks |
| `/silicon-entitlements` | Dump any binary's entitlements |

---

*Tested on Mac mini M4 16GB, macOS 26.3.1 Tahoe. Private APIs may change between versions.*
