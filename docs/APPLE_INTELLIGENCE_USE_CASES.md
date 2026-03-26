# Apple Intelligence — Complete Use Case Map

100+ Apple Intelligence use cases extracted from the UAF (Unified Action Framework) configuration on macOS 26.3.1.

Each use case maps to: a model adapter (LoRA), safety filters (input + output), and language support configuration.

Source: `/System/Library/AssetsV2/com_apple_MobileAsset_UAF_Siri_PlatformAssets/.../UsageAliases/`

## Architecture

```
Use Case Config (plist)
  → Model Catalog (which base model + LoRA adapter to load)
  → Safety Filters (input deny + output deny + embedding deny)
  → Language Support (per-language enablement)
  → FM.Overrides (runtime config adjustments)
```

Each use case has a `.draft` variant (for speculative/fast generation) and a production variant.

## Writing & Text Composition (17 use cases)

| Use Case | What it does |
|----------|-------------|
| `textComposition.ConciseTone` | Rewrite text in concise tone |
| `textComposition.FriendlyTone` | Rewrite text in friendly tone |
| `textComposition.ProfessionalTone` | Rewrite text in professional tone |
| `textComposition.MagicRewrite` | Creative text rewriting |
| `textComposition.ProofreadingReview` | Grammar, spelling, punctuation check |
| `textComposition.BulletsTransform` | Convert text to bullet points |
| `textComposition.TablesTransform` | Convert text to table format |
| `textComposition.TakeawaysTransform` | Extract key takeaways |
| `textComposition.MailReplyLongFormBasic` | Generate email replies (long form) |
| `textComposition.MailReplyLongFormRewrite` | Rewrite email replies |
| `textComposition.MailReplyQA` | Q&A-style email reply |
| `textComposition.MessagesReply` | iMessage smart replies |
| `textComposition.MessagesUserRequest` | Custom iMessage composition |
| `textComposition.OpenEndedTone` | Open-ended tone adjustment |
| `textComposition.OpenEndedSchema` | Schema-based text generation |
| `textComposition.TextExpert` | General text expertise |
| `textComposition.SuperAutofill` | Smart form autofill |

All Writing Tools tasks use: `instruct_3b.base` + task-specific LoRA adapter, with `safety_deny.input.<task>` and `safety_deny.output.<task>` filters.

## Summarization (2 use cases)

| Use Case | What it does |
|----------|-------------|
| `summarization.generativePlayground` | General text summarization |
| `summarization.textAssistant` | Assistant-style summarization |

## Classification (5 use cases)

| Use Case | What it does |
|----------|-------------|
| `classification.classifyMailMessage` | Categorize emails (priority, type) |
| `classification.classifyTextMessage` | Categorize text messages |
| `classification.classifyTextMessageThread` | Categorize message threads |
| `classification.classifyUserNotification` | Prioritize notifications |
| `Automated.Reminders.AutoCategorization` | Auto-categorize reminders |

## Text Understanding (3 use cases)

| Use Case | What it does |
|----------|-------------|
| `textUnderstanding.TextEventExtraction` | Extract events (dates, times, locations) |
| `textUnderstanding.TextPersonExtraction` | Extract person names and references |
| `textUnderstanding.StructuredExtraction` | General structured data extraction |

## Visual Generation / Image Playground (16 use cases)

| Use Case | Style |
|----------|-------|
| `VisualGeneration.GenerativePlayground:illustration:illustration` | Standard illustration |
| `VisualGeneration.GenerativePlayground:illustration:personalized_illustration` | Personalized illustration |
| `VisualGeneration.GenerativePlayground:illustration:style_scribble` | Scribble-style illustration |
| `VisualGeneration.GenerativePlayground:sketch:sketch` | Standard sketch |
| `VisualGeneration.GenerativePlayground:sketch:personalized_sketch` | Personalized sketch |
| `VisualGeneration.GenerativePlayground:sketch:style_scribble` | Scribble sketch |
| `VisualGeneration.GenerativePlayground:emoji:emoji` | Genmoji |
| `VisualGeneration.GenerativePlayground:emoji:personalized_emoji` | Personalized Genmoji |
| `VisualGeneration.GenerativePlayground:animation:animation` | Animated image |
| `VisualGeneration.GenerativePlayground:animation:personalized_animation` | Personalized animation |
| `VisualGeneration.GenerativePlayground:animation:style_scribble` | Scribble animation |
| `VisualGeneration.GenerativePlayground:messages-background:messages_backgrounds` | iMessage backgrounds |
| `VisualGeneration.KeyboardEmojiGenerator:emoji:emoji` | Keyboard emoji generation |
| `VisualGeneration.KeyboardEmojiGenerator:emoji:personalized_emoji` | Personalized keyboard emoji |
| `VisualGeneration.MessagesBackgrounds` | Message background generation |
| `com.apple.PaperKit.ImageGeneration` | PaperKit image generation |

## Visual Intelligence (3 use cases)

| Use Case | What it does |
|----------|-------------|
| `com.apple.VisualIntelligence.Tamale` | Visual intelligence (point camera at things) |
| `com.apple.VisualIntelligence.structuredExtraction.addToCalendar` | Extract events from camera |
| `com.apple.VisualIntelligence.structuredExtraction.addToContacts` | Extract contacts from camera |

## Siri & Assistant (6 use cases)

| Use Case | What it does |
|----------|-------------|
| `com.apple.Siri.Planner` | Siri task planning |
| `com.apple.siri.assistant.language` | Siri language understanding |
| `com.apple.siri.assistant.hybrid.language` | Hybrid (on-device + server) |
| `com.apple.siri.assistant.legacy.language` | Legacy Siri compatibility |
| `com.apple.siri.assistant.assistantengine.language` | Core assistant engine |
| `com.apple.siri.nl.system.language` | Natural language system |

## Photos & Memories (4 use cases)

| Use Case | What it does |
|----------|-------------|
| `memoryCreation.GlobalTraits` | Generate personality traits for memories |
| `memoryCreation.AssetCuration` | Curate photos for memories |
| `memoryCreation.Storyteller` | Generate memory narratives |
| `memoryCreation.QueryUnderstanding` | Understand photo search queries |

## Fitness Intelligence (6 use cases)

| Use Case | What it does |
|----------|-------------|
| `FitnessIntelligence.WorkoutVoice.Breakthrough` | Celebrate workout breakthroughs |
| `FitnessIntelligence.WorkoutVoice.Breakthrough.migration` | Migration variant |
| `FitnessIntelligence.WorkoutVoice.Companion.Breakthrough` | Companion breakthrough |
| `FitnessIntelligence.WorkoutVoice.Companion.Outro` | Workout companion outro |
| `FitnessIntelligence.WorkoutVoice.Outro` | Workout outro |
| (various migration variants) | Backward compatibility |

## Shortcuts & Automation (4 use cases)

| Use Case | What it does |
|----------|-------------|
| `com.apple.Shortcuts.AskAFMAction3B` | Shortcuts "Ask Apple Intelligence" (3B on-device) |
| `com.apple.Shortcuts.AskMontaraAction` | Shortcuts "Ask Montara" (server model?) |
| `deviceexpert.ContextualRewrite` | Device-context-aware rewriting |
| `com.apple.mlda.AutoTagger` | Automatic content tagging |

## Speech & Dictation (2 use cases)

| Use Case | What it does |
|----------|-------------|
| `asr.Dictation.DescribeYourEdit` | "Describe your edit" for dictation |
| `asr.fullPayloadCorrection` | Full dictation correction |

## Other (5 use cases)

| Use Case | What it does |
|----------|-------------|
| `accessibility.magnifier` | Magnifier accessibility feature |
| `com.apple.FoundationModels` | Public FoundationModels API (developer access) |
| `com.apple.vision.adt` | Vision ADT (Advanced Detection & Tracking?) |
| `financialInsights.AppleIntelligence` | Financial insights |
| `journaling.FollowUpPrompts` | Journal follow-up suggestions |
| `com.apple.photos.semanticSearch.test` | Photo semantic search (test) |
| `com.apple.coremotion.fitness` | Core Motion fitness intelligence |

## Safety Architecture

Every use case has safety filters at two levels:

### Input Safety
- `safety_deny.input.<task>.base.generic` — Blocks unsafe input
- `safety_deny.input.<task>.visual_prompt.generic` — Blocks unsafe visual prompts (for image generation)

### Output Safety
- `safety_deny.output.<task>.base.generic` — Blocks unsafe output
- `safety_deny.output.<task>.visual_prompt.generic` — Blocks unsafe visual output

### Global Safety
- `safety_embedding_deny.all.generic` — Embedding-level content filtering
- `safety.disabledusecases.generic` — Disabled use cases list
- `prompt_allow_list.input.delta_lexicon.generic` — Allowed prompt vocabulary

## Language Support

Each use case is independently enabled per language. Observed languages in configs:
`da` (Danish), `de` (German), `en` (English), `es` (Spanish), `fr` (French), `it` (Italian), `ja` (Japanese), `ko` (Korean), `nl` (Dutch), `pt` (Portuguese), `sv` (Swedish), `zh` (Chinese)

## System Prompts

The actual system prompt text (instructions to the 3B model) is NOT stored as plain text on disk. It is either:
1. Baked into the LoRA adapter weights during training
2. Delivered via FM.Overrides MobileAsset downloads (protected storage, requires entitlements)
3. Generated at runtime by the `PromptGenerator` classes in PhotosIntelligence and GenerativeExperiences frameworks

The configs we found define WHICH models and filters to activate, not the prompt text itself.

## FoundationModels — The Public API

`com.apple.FoundationModels` is the use case for the new public FoundationModels framework (announced WWDC25). It uses:
- `instruct_3b.base` + `instruct_3b.base_adapter`
- `instruct_3b.fm_api_generic` (general purpose adapter)
- `instruct_3b.fm_api_content_tagger` (content tagging)
- `instruct_3b.third_party` (third-party app usage)

This is how developers will access Apple's on-device LLM in their apps.
