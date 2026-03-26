# macOS 26 Tahoe — New APIs Not Present in macOS 15

Discovered via runtime introspection on macOS 26.3.1 (Build 25D771280a), Apple M4.

These frameworks and classes are likely new to macOS 26. They did not exist in public documentation as of macOS 15 Sequoia.

## 1. Image Playground / Genmoji Engine (184 classes)

`ImagePlaygroundInternal.framework` — Apple's on-device image generation system.

### Key Classes

| Class | Function |
|-------|----------|
| `GenmojiComposingViewModel` | Genmoji creation interface |
| `EmojiImageRenderer` / `EmojiImageProvider` | Custom emoji rendering |
| `_ConditioningImageFactory` | Conditioning images (ControlNet-like) |
| `ExternalProviderImageGenerator` | External generation provider support |
| `PromptConceptAnalyzer` | Prompt analysis for generation |
| `LoadGPRecipeInContextOperation` | Load generation "recipes" |
| `MagicViewModel` | "Magic" editing interface (Clean Up / generative fill) |
| `LayerRenderer` | Multi-layer rendering |
| `StickerSaveAnimation` | Sticker creation/export |
| `ImagePlaygroundStyleSource` | Style selection |
| `SessionUndoManager` | Undo/redo for generation |
| `DescriptionFieldWithSuggestionsViewModel` | Prompt suggestions |
| `ServiceSideConnectionManager` | Backend service connection |
| `CuratedPromptsManager` | Curated prompt library |

### Architecture

Image Playground uses a recipe-based system (`LoadGPRecipeInContextOperation`) with conditioning images (`_ConditioningImageFactory`). Supports external providers (`ExternalProviderImageGenerator`), suggesting a plugin architecture for image generation backends.

## 2. OSIntelligence — Battery & Power Prediction AI (27 classes)

`OSIntelligence.framework` — ML-based power management.

### Key Classes

| Class | Function |
|-------|----------|
| `_OSBatteryDrainPredictor` | Predicts battery drain rate |
| `_OSInactivityPredictionClient` | Predicts user inactivity windows |
| `_OSIAutoLPMManager` | Automatic Low Power Mode activation |
| `_OSICLPCInterface` | Direct interface to CLPC performance controller |
| `_OSChargingPredictorOutput` | Predicts charging time |
| `_OSLastLockPredictorOutput` | Predicts when device will be locked |
| `_OSIBatteryLifeManager` | Battery life management |
| `_OSIBLMitigation` | Battery life mitigation strategies |
| `CPUEnergySnapshot` | CPU energy snapshots for prediction |

### How It Works

OSIntelligence predicts user behavior (inactivity, locking, charging) and proactively adjusts power management via the CLPC interface. This is the AI behind "Optimized Battery Charging" and automatic Low Power Mode.

## 3. IntelligenceFlow — Siri/AI Task Orchestration (29+ classes)

`IntelligenceFlow.framework` + `IntelligenceFlowRuntime.framework` — The orchestration layer for all Apple Intelligence operations.

### Key Classes

| Class | Function |
|-------|----------|
| `SessionClient` / `SessionDebugger` | Conversation session management |
| `SnippetStreamingClient` | Streaming responses (like ChatGPT) |
| `TranscriptEntityQueryingClient` | Query entities in conversation history |
| `ToolboxClient` | Access to AI tools |
| `TurnManager` | Conversation turn management |
| `QueryDecoration` | Query enrichment/preprocessing |
| `ResponseOverrideMatcher` | Response filtering/override |
| `SpeechHandler` | Speech I/O handling |

### Architecture

IntelligenceFlow is the middleware between Siri's UI and the AI backends. It manages conversation sessions, streams responses, queries conversation history for context, and coordinates tool execution. The `TurnManager` suggests multi-turn conversation support.

## 4. ComputeSafeguards — Intelligent Compute Throttling (28 classes)

`ComputeSafeguards.framework` — Scenario-based compute restriction system.

### Key Classes

| Class | Function |
|-------|----------|
| `CSScenarioManager` | Manages active compute scenarios |
| `CSScenario` | Defines a compute scenario (e.g., "thermal throttle") |
| `CSCPUTimeRestriction` | CPU time limits per scenario |
| `CPUEnergySnapshot` | Energy consumption snapshots |
| `CSProcessManager` / `CSProcess` | Per-process tracking |
| `CSIssueDetector` | Detects compute issues (thermal, battery) |
| `CSRestrictionFactory` | Creates restrictions based on conditions |
| `CSContextStore` | Stores scenario context |
| `CSDaemon` | Background daemon |

### How It Works

ComputeSafeguards runs scenarios that define compute restrictions. When conditions are met (thermal pressure, low battery, sustained load), restrictions are applied to processes. This is more granular than the existing thermal management — it's scenario-driven and per-process.

## 5. WritingTools — Smart Reply & Text Rewriting (6 classes)

`WritingTools.framework` — The Writing Tools system that powers Proofread, Rewrite, and Smart Reply.

### Key Classes

| Class | Function |
|-------|----------|
| `WTSmartReplyConfiguration` | Smart Reply config with `baseResponse`, `entryPoint`, `inputContextHistory` |
| `WTContext` | Text context with `attributedText`, `range`, `smartReplyConfig`, `uuid` |
| `WTBSCompatibleAttributedString` | BlastDoor-safe attributed string |

## 6. VideoIntelligence — Video Understanding (20 classes)

`VideoIntelligence.framework` — On-device video analysis and captioning.

### Key Classes

| Class | Function |
|-------|----------|
| `VideoCaptioningModel` | Video captioning ML model |
| `VideoCaptioningProvider` | Captioning service |
| `VideoCaptioningUseCase` | Use case definitions |
| `AssetRegistry` | Model asset management (MobileAsset + bundle sources) |
| `PixelRotationSession` / `PixelTransferSession` | Video frame processing |

## 7. GenerativeFunctions — Streaming JSON Parser (4 classes)

`GenerativeFunctions.framework` — Infrastructure for streaming generative responses.

| Class | Function |
|-------|----------|
| `JsonTokenStream` | Token-level JSON streaming |
| `JsonStreamParser` | Incremental JSON parser |
| `StreamingObjectTokenStreamProcessor` | Processes streaming objects |

## 8. Other Notable New Frameworks

| Framework | Classes | Purpose |
|-----------|---------|---------|
| `GenerativePartnerService` | 5 | Third-party generative AI integration |
| `LLMCache` | 4 | LLM response caching (vector DB, annotations) |
| `AppleIntelligenceReporting` | 5 | AI usage analytics/reporting |
| `iCloudMailAssistant` | 61 | AI-powered email management |
| `PersonalIntelligenceCore` | 9 | Personal context for AI |
| `MomentsIntelligence` | 3 | Photo memories AI |
| `SiriInteractive` | 7 | Interactive Siri UI with snippet hosting |
| `LiveSpeechServices` | 3 | Real-time speech services |
| `DoubleAgent` | 2 | Unknown (AppleDouble file parser + manager) |

## 9. M5 GPU Firmware Already Present

Kernel extensions for unreleased M5 chip family found:

```
AGXG17G.kext    → M5 (standard)     AGXAcceleratorG17G
AGXG17P.kext    → M5 Pro             AGXAcceleratorG17P
AGXG17X.kext    → M5 Max             AGXAcceleratorG17X
AGXMetalG17G/P/X.bundle → Metal shader bundles per variant
AGXFirmwareKextG17G/P/XRTBuddy.kext → Firmware loaders
```

The "P" suffix (Pro) is new — previous generations only had G (standard) and X (Pro/Max). This suggests M5 Pro has sufficiently different GPU hardware to warrant its own firmware, separate from M5 Max.

## Total New API Surface

| Category | Frameworks | Classes |
|----------|-----------|---------|
| Image Generation | 1 | 184 |
| Power AI | 1 | 27 |
| AI Orchestration | 6 | ~80 |
| Compute Management | 1 | 28 |
| Writing Tools | 1 | 6 |
| Video Intelligence | 1 | 20 |
| Generative Functions | 3 | ~15 |
| Other Intelligence | 8 | ~95 |
| **Total** | **~22** | **~455** |

~455 new classes across ~22 frameworks dedicated to Apple Intelligence in macOS 26.
