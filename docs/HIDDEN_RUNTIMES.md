# Hidden Runtimes & ML Infrastructure in macOS 26

Deep scan of intriguingly-named private frameworks revealed Apple has embedded complete ML runtimes, JIT compilers, and Python interpreters inside macOS.

## Morpheus (81 classes) — MLX + Python Runtime

Apple embedded an **MLX runtime with a Python interpreter** inside macOS. This is likely what executes Apple Intelligence models internally.

### Key Classes

| Class | Function |
|-------|----------|
| `MLXArray` | MLX array type (Apple's ML framework) |
| `MLXFast.MLXFastKernel` | Optimized MLX compute kernels |
| `PyCustomClass` | Python custom class support |
| `PyObjectInstance` | Python object instances |
| `PyTypes` | Python type system |
| `PyBuiltInClass` | Python built-in classes |
| `PyJsonModule` | Python JSON module |
| `PyLoggingModule` | Python logging |
| `PyLogger` | Logger implementation |
| `ModuleResolver` | Python module resolution |
| `Device` | Compute device abstraction |
| `ErrorHandler` / `ErrorBox` | Error handling |
| `new_mlx_closure` / `new_mlx_kwargs_closure` | MLX function closures with kwargs |

### Implications

Morpheus bridges Python and MLX at the system level. The closure classes (`new_mlx_closure`, `new_mlx_kwargs_closure`) suggest it can execute Python-defined MLX functions natively. This may be how Apple converts Python ML models to on-device execution without requiring a full Python install.

## XOJIT (10 classes) — LLVM-based JIT Compiler

A **Just-In-Time compiler** built on LLVM's ORC (On-Request Compilation) framework.

### Key Classes

| Class | Function |
|-------|----------|
| `XOJIT` | Main JIT compiler |
| `LLVMMemoryBuffer` / `MemoryBuffer` | LLVM memory management |
| `ORCRuntimeBridge` | Bridge to LLVM ORC runtime |
| `ReplacementManager` | Hot-swaps compiled code |
| `MachOHeaderOptions` | Mach-O binary generation |
| `JITDylib` / `JITDylibRef` | JIT dynamic libraries |
| `SymbolTableStream` | Symbol resolution |
| `XOJITError` | Error handling |

### XOJITExecutor (3 classes)

| Class | Function |
|-------|----------|
| `XOJITExecutor` | Executes JIT-compiled code |
| `ActiveRunProgram` | Currently executing program |
| `ActiveRunWrapper` | Execution wrapper |

### Implications

XOJIT compiles code on-the-fly using LLVM. The `ReplacementManager` suggests it can hot-swap compiled functions (useful for ML graph optimization). `MachOHeaderOptions` means it generates native ARM64 Mach-O code at runtime.

## ODIE (27 classes) — ML Model Virtual Machine

A **virtual machine for executing ML models** with its own register bank and call table.

### Key Classes

| Class | Function |
|-------|----------|
| `Model` | ML model representation |
| `NDArray` | N-dimensional array |
| `RegisterBank` | Register file (VM execution state) |
| `CallTable` | Function dispatch table |
| `InternalSymbolResolver` | Symbol resolution |
| `Verifier` | Model verification |
| `FlatBufferBuilder` | FlatBuffer serialization |
| `ByteBuffer` / `_InternalByteBuffer` | Memory management |

### Implications

ODIE is a bytecode VM for ML models. `RegisterBank` + `CallTable` = a virtual ISA. This is likely the execution engine behind CoreML's "mlProgram" model type. Models are compiled to ODIE bytecode and executed in this VM, which can dispatch to CPU (Accelerate/SME2), GPU (Metal), or ANE depending on the operation.

## TuriCore (67 classes) — On-Device Training

Apple's on-device model training infrastructure, from their 2016 Turi acquisition.

### Key Classes

| Class | Function |
|-------|----------|
| `TCModelTrainerBackendGraphs` | Training computation graphs |
| `TCModelTrainerBackendGraphsWithSplitLoss` | Training with distributed loss |
| `TCMLComputeObjectDetectorDescriptor` | Object detection model training |
| `TCMPSLabeledImage` | Training data (labeled images) |
| `TCMPSImageAnnotation` | Image annotations |
| `TCMPSRandomCropAugmenter` | Data augmentation: random crop |
| `TCMPSHorizontalFlipAugmenter` | Data augmentation: flip |
| `TCMPSHueAdjustAugmenter` | Data augmentation: color |
| `TCMPSColorControlAugmenter` | Data augmentation: contrast/brightness |
| `TCMPSResizeAugmenter` | Data augmentation: resize |
| `TCMPSRandomPadAugmenter` | Data augmentation: padding |
| `TCComputeDevice` / `TCComputeDeviceManager` | Compute device management |
| `TCPreferences` | Training preferences |

### Implications

Apple has a complete on-device training pipeline: labeled data → augmentation → training graphs → model. This is what powers Create ML and on-device personalization (e.g., Photos learning to recognize your face better over time).

## CVNLP (40 classes) — Video Captioning with CTC

Computer Vision + NLP: generates text captions from video using CTC (Connectionist Temporal Classification) beam search.

### Key Classes

| Class | Function |
|-------|----------|
| `CVNLPVideoCaptioningModel` | Video → text caption model |
| `CVNLPCTCBeamState` | CTC beam search state |
| `CVNLPCTCTextDecodingPath` | CTC decoding path |
| `CVNLPTextDecoder` | Text decoder |
| `CVNLPTextDecodingContext` | Decoding context |
| `CVNLPDecodingLexicon` | Vocabulary/lexicon |
| `CVNLPTokenIDConverter` | Token ↔ ID conversion |
| `CVNLPTextDecodingResult` | Decoding result |
| `CVNLPCaptionRuntimeExcludeGenderTrigger` | Gender-neutral caption filter |
| `CVNLPPerformance` / `CVNLPPerformanceResult` | Performance metrics |

## DeepThought (131 classes) — Siri Usage Analytics

Named after the Hitchhiker's Guide supercomputer. Tracks how and when users interact with Siri.

### Key Capabilities

- `SiriPenetrationRateCalculator` — measures Siri adoption rate
- `FeaturizedConversationDataProvider` — extracts features from conversations
- `FeaturizedBiomeDataProvider` — extracts features from user behavior (Biome)
- Multiple reporters: CoreAnalytics, Biome, SELF, JSON logging
- Event filters: SMS, phone calls, reminders, etc.

## Other Discoveries

| Framework | Classes | What it is |
|-----------|---------|-----------|
| `Engram` (20) | `ENGroup`, `ENParticipant`, `ENCypher_AES128` | Encrypted group communication (E2E?) |
| `Synapse` (67) | `SYDocument`, `SYBacklinkIndicator`, `SYDocumentWorkflows` | Document linking/workflow system |
| `Sage` (30) | Not scanned deep | Unknown (name suggests wisdom/advice) |
| `PromptKit` (1) | `AnyGenerationGuides` | Generation guidance system |
| `SentencePieceInternal` (1) | — | SentencePiece tokenizer (used in LLMs) |
| `Dendrite` (15) | — | Neural connection metaphor |
| `Koa` (43) | — | Unknown |
| `C2` (247) | — | Cloud/Command & Control? Large framework |
| `Anvil` (14) | — | Unknown |
| `Chirp` (3) | — | Audio/communication |

## The ML Execution Stack

Putting it all together, Apple's ML execution stack on macOS 26:

```
Python/MLX model definition
    ↓
Morpheus (Python → MLX bridge)
    ↓
XOJIT (JIT compile compute graphs via LLVM ORC)
    ↓
ODIE VM (execute model bytecode)
    ├── RegisterBank + CallTable
    ├── → CPU (Accelerate/SME2)
    ├── → GPU (Metal/MPS)
    └── → ANE (_ANEInMemoryModel)
    ↓
VectorSearch (store embeddings)
    ↓
Apple Intelligence output
```

This is the full pipeline from model definition to execution to storage, all on-device.
