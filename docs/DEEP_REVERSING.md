# Deep Reverse Engineering Findings

Additional discoveries beyond the initial API scan. Covers CoreML model extraction, ANE instruction format, Metal shader libraries, PCC protocol, and XPC daemon architecture.

## 1. CoreML Models on System — 614 Models Found

Apple ships 614 compiled CoreML models (`.mlmodelc`) across system frameworks and assets.

### Largest Neural Network Models

| Size | Location | Architecture | Function |
|------|----------|-------------|----------|
| 68 MB | Siri TTS `p2a_single_prompt_decoder` | mlProgram | Text-to-Speech decoder (F32, 476x519 input) |
| 65 MB | MediaAnalysis `mubb_md7` | mlProgram | Video scene understanding |
| 64 MB | CoreSceneUnderstanding `text_md7_6bit_ctx_512_77` | mlProgram | CLIP-like text encoder (FP16, 6-bit quantized, 512 token context) |
| 59 MB | Siri TTS `p2a_single_prompt_encoder` | mlProgram | TTS prompt encoder |
| 45 MB | Siri TTS `anetec_decoder_streaming_inference` | mlProgram | Streaming TTS decoder |
| 20 MB | CoreSceneUnderstanding `token_md7_6bit` | mlProgram | Token encoder for visual search |
| 11 MB | SoundAnalysis `SNLanguageAlignedAudioEncoder` | mlProgram | Audio → 512-dim embedding |
| 9 MB | Siri Understanding `bert` | mlProgram | BERT for Siri input representations |

### Apple's CLIP Architecture

`text_md7_6bit_ctx_512_77.mlmodelc`:
```
Input:  token_embed  [1, 512, 256]  FP16
        indices      [1]            FP16
Output: text_embed   [1, 512]       FP16  (final text embedding)
        spatial_embed [1, 512, 768] FP16  (per-token spatial features)
        hidden_embed [1, 768]       FP16  (hidden state)
```
Custom CLIP variant with 6-bit quantization and 512-token context. Used by Spotlight/Photos for visual search.

### Apple's Audio Encoder

`SNLanguageAlignedAudioEncoder.mlmodelc`:
```
Input:  specgram  [1, 197, 64]  FP16  (mel spectrogram)
Output: embedding [1, 512]      FP16  (language-aligned audio embedding)
```
Maps audio to the same 512-dim embedding space as the text CLIP encoder. This is how Siri matches spoken words to visual concepts.

### Model Categories

| Category | Count | Examples |
|----------|-------|---------|
| Intelligence/Prediction | ~50 | Entity relevance, transport mode, app prediction |
| Siri Understanding | ~30 | BERT, slot filling, intent classification |
| Siri TTS | ~20 | Prompt encoder/decoder, streaming inference |
| Vision/Scene | ~40 | CLIP encoder, image captioning, video captioning |
| Translation | ~15 | Language detection, confidence models |
| Sound Analysis | ~10 | Audio encoder, sound classification |
| Face/Body | ~30 | Face attributes, body pose, segmentation |
| NLP | ~20 | Sentiment, NER, grammar correction |
| Others | ~400 | Spotlight search, bias estimation, widget prediction |

## 2. ANE Instruction Format

### File Types

| Extension | Format | Purpose |
|-----------|--------|---------|
| `.espresso.net` | Espresso network graph | Defines network topology (layers, connections) |
| `.espresso.shape` | Shape metadata | Tensor dimensions for each layer |
| `.espresso.weights` | Binary weights | Raw weight data |
| `.hwx` | ANE hardware bytecode | Compiled instructions for ANE execution |

### CPU vs ANE Dual Compilation

Apple compiles the same model for both backends:
```
unilm_joint_cpu.espresso.net     ← CPU execution path
unilm_joint_ane.espresso.net     ← ANE execution path
cpu_embeddings.espresso.net      ← CPU embeddings
ane_embeddings.espresso.net      ← ANE embeddings
```
The runtime selects CPU or ANE based on availability and load.

### HWX Models (ANE Hardware Instructions)

Found in:
- `AppleISPEmulator.framework` — ISP (Image Signal Processor) models at various resolutions (1080p, 1552x1552, etc.)
- `VideoProcessing.framework` — Frame enhancement CNNs compiled for different chip generations (H13=M1, H14=M2, H16=M4)

File naming pattern: `cnn_frame_enhancer_{resolution}.{chip_gen}.espresso.hwx`

## 3. Metal Shader Libraries — 265 Files

### Largest Shader Libraries

| Size | Framework | Content |
|------|-----------|---------|
| 154 MB | QuartzCore | Window compositing, Core Animation |
| 100 MB | CoreImage (ubershader) | All CI filter implementations |
| 76 MB | CoreImage (uberwrapper) | CI filter wrappers |
| 62 MB | MPS (4 libs total) | MatMul, convolution, attention, FFT kernels |
| 36 MB | MXI | Mixed-precision inference kernels |
| 28 MB | Espresso | ML compute kernels (what CoreML uses internally) |
| 14 MB | RenderBox | 3D rendering |
| 12 MB | ShaderGraph | Reality Composer shaders |
| 5.7 MB | Metal (BVH) | Ray tracing acceleration structure builder |

The Espresso `default.metallib` (28 MB) is the most relevant — it contains Apple's optimized GPU kernels for ML operations that CoreML dispatches to.

## 4. XPC Daemon Architecture

### Running Apple Intelligence Daemons

| PID | Process | Mach Service | Function |
|-----|---------|-------------|----------|
| 999 | siriinferenced | `com.apple.siriinferenced` | Main inference, suggestions |
| 5953 | intelligenceplatformd | — | Platform orchestration |
| 39505 | IntelligencePlatformComputeService | — | Compute dispatch (XPC) |
| 966 | intelligencecontextd | — | Context flow |
| 956 | siriknowledged | — | Knowledge base |
| 53293 | ModelCatalogAgent | `com.apple.modelcatalog.subscriptions` | Model catalog management |
| 53286 | generativeexperiencesd | — | Generative experiences |
| 914 | HostInferenceProviderService | — | Model execution (user `_modelmanagerd`) |

### Model Catalog Agent

Runs as a subscription-based Mach service. Other daemons subscribe to model availability notifications via `com.apple.modelcatalog.subscriptions`. When models are downloaded or updated via MobileAsset, subscribers get notified.

## 5. Private Cloud Compute Protocol

### Feature Flags

```
featureUsageAnalyzer:        FeatureComplete
forceTrustedProxyProtocol:   FeatureComplete
trustedProxyProtocol:        FeatureComplete
```

### PCC Daemon Entitlements

`privatecloudcomputed` has:
- `com.apple.privatecloudcomputed` — main PCC entitlement
- `com.apple.transparency.privateCloudCompute` — attestation/transparency
- `com.apple.private.network.socket-delegate` — network socket delegation
- `com.apple.private.network.system-token-fetch` — system token retrieval
- `com.apple.private.cloudtelemetry` — telemetry reporting

### PCC Client Classes

| Class | Role |
|-------|------|
| `PrivateCloudCompute.TC2Client` | Trusted Cloud Compute v2 client |
| `PrivateCloudCompute.TrustedCloudComputeClient` | Attestation client |
| `PrivateCloudCompute.XPCWrapper` | XPC communication layer |
| `PCCServerEnvironment` | Server environment config |
| `PrivateMLClient.PrivateMLClientCloudComputeConnection` | Inference connection |

### Inference Protocol

`PrivateMLClient` uses protobuf-like structures:
- `Apple_Cloudml_Inference_Tie_GenerateRequest` — request with model config
- `Apple_Cloudml_Inference_Tie_GenerateResponse` — response with debug info
- `Apple_Cloudml_Inference_Tie_ModelConfig` — model selection

## 6. Third-Party App Analysis

### Brave Browser

Uses `Accelerate.framework`, `IOKit`, `LocalAuthentication`, `CoreImage`, `OpenGL`.
Entitlements include VPN API access (`com.apple.developer.networking.vpn.api`), USB, Bluetooth, camera, location.

### Common Patterns

- Electron apps (Claude, Cursor): minimal system integration, JIT entitlement for V8
- Native Apple apps: 30-57 private frameworks, deep system integration
- Third-party native apps: primarily public frameworks, occasional private framework usage for competitive advantage
