# Apple's Built-in Vector Database & Embedding System

macOS 26 has a native vector database, embedding model, and semantic search system built into the OS. This is the infrastructure behind Apple Intelligence's on-device document understanding — RAG (Retrieval Augmented Generation) without cloud.

Discovered via ObjC runtime introspection on macOS 26.3.1, Apple M4.

## Architecture

```
User documents/emails/notes/messages
    ↓
SPEmbeddingModel (on-device embedding model)
    ↓
SPEmbeddingTokenizer → SPEmbeddingResult (vector)
    ↓
VDBVectorDatabase (SQLite-backed vector store)
    ↓
VectorQueryProcessor (semantic search)
    ↓
Apple Intelligence / Spotlight / Siri
```

## VectorSearch.framework (31 classes)

The vector database layer. Built on SQLite with vector extensions.

### Core Database

| Class | Properties/Methods | Function |
|-------|-------------------|----------|
| `VDBVectorDatabase` | — | Vector database implementation |
| `VectorDatabasePool` | — | Connection pooling |
| `VectorDatabaseClient` | — | Client interface |
| `Index` | — | Vector index (likely HNSW or similar) |
| `DeltaManager` | — | Incremental index updates |
| `SQLPreparedStatement` | — | SQL layer (SQLite) |
| `SQLDatabasePointer` | — | Database handle |
| `SQLExpressionEvaluator` | — | Expression evaluation |

### Search & Query

| Class | Properties | Function |
|-------|-----------|----------|
| `VectorQueryProcessor` | — | Processes vector similarity queries |
| `VSKSearchResult` | 5 properties | Search result with score/metadata |
| `VSKFilter` | — | Filter search results |
| `VSKDisjunctiveFilter` | 2 properties | OR-combined filters |
| `VSKPagination` | 2 properties | Paginated results |
| `VSKStatistics` | 6 properties | Query statistics |

### Storage

| Class | Function |
|-------|----------|
| `AttributeStore` | Stores document attributes alongside vectors |
| `AssetStore` | Stores assets (documents, images) |
| `VSKAsset` | Individual asset (5 properties) |
| `VSKAttribute` | Document attribute |
| `VSKDatabaseValue` | Typed database value |
| `VSKColumnType` | Column type definitions |
| `VSKConfig` | Database configuration |
| `VSKClient` | High-level client |

### Text Processing

| Class | Function |
|-------|----------|
| `UnicodeWrapperTokenizer` | Unicode-aware tokenizer |
| `Unicode61WrapperTokenizerDataReference` | Tokenizer data |

## SpotlightEmbedding.framework (5 classes)

The embedding model that converts text to vectors.

| Class | Properties | Methods | Function |
|-------|-----------|---------|----------|
| `SPEmbeddingModel` | 1 | 18 instance + 3 class | The embedding model itself |
| `SPEmbeddingTokenizer` | 1 | 7 instance + 2 class | Text tokenization for embedding |
| `SPEmbeddingResult` | 6 | 9 instance + 1 class | Embedding vector output |
| `SPTextInput` | 4 | 13 instance | Text input for embedding |
| `SPEmbeddingTailspinDumper` | 1 | 8 instance + 5 class | Diagnostics/debugging |

`SPEmbeddingModel` has 18 instance methods — this is a substantial model interface, likely wrapping a CoreML model that generates embeddings on-device.

## EmbeddingService.framework (2 classes)

Centralized embedding service accessible system-wide.

| Class | Properties | Methods | Function |
|-------|-----------|---------|----------|
| `QUEmbeddingService` | 3 | 13 instance + 5 class | System-wide embedding service |
| `QUEmbeddingOutput` | 9 | 11 instance | Embedding output with rich metadata |

`QUEmbeddingOutput` has 9 properties — likely includes the vector, confidence, token count, model version, and other metadata.

## SemanticDocumentManagement.framework (14 classes)

Semantic indexing for Apple Help and documents.

| Class | Function |
|-------|----------|
| `HelpSDMIndex` | Semantic document index |
| `HelpSDMIndexBuilder` | Builds semantic indices |
| `HelpSDMModel` | Semantic model |
| `SDMQuery` | Semantic query |
| `SDMQueryResult` | Query result |
| `LSMDocMetadata` | Document metadata |
| `HelpTokenizer` | Base tokenizer |
| `HelpTokenizerRomanBasedNgrams` | N-gram tokenizer for Roman scripts |
| `HelpTokenizerItalian` / `HelpTokenizerFrench` | Language-specific tokenizers |
| `HelpTokenizerNonRomanNgrams` | N-gram tokenizer for CJK etc. |

## How Apple Uses This

### Spotlight Semantic Search
When you search in Spotlight, it's no longer just keyword matching. Documents are embedded as vectors, and search queries are also embedded, enabling semantic similarity matching. "Find my tax documents" can find files even if they don't contain the word "tax".

### Apple Intelligence Context
When Apple Intelligence needs to reference your documents (e.g., "summarize my recent emails about the project"), it queries the vector database to find semantically relevant content, then passes it to the LLM as context. This is on-device RAG.

### Siri Knowledge
Siri can now answer questions about your personal data by querying the vector database for relevant context before generating a response.

## Implications

1. **Apple built RAG into the OS** — no third-party vector database needed
2. **Everything is on-device** — vectors never leave the Mac
3. **SQLite-backed** — the vector database uses SQLite with extensions, not a separate database engine
4. **System-wide service** — any app could potentially use `EmbeddingService` and `VectorSearch`
5. **Multi-language** — tokenizers for Roman, Italian, French, and non-Roman (CJK) scripts

## Also Discovered (Unexplored)

Massive frameworks we identified but didn't deep-scan:

| Framework | Classes | What it likely does |
|-----------|---------|-------------------|
| AppPredictionInternal | 999 | Predicts which app you'll open next |
| BiomeStreams | 468 | User behavior/activity tracking pipeline |
| PersonalizationPortraitInternals | 331 | User personalization model |
| CoreSuggestionsInternals | 362 | Context-aware suggestions |
| HealthDaemon | 1,408 | Health data processing (biggest framework on the system) |
| WorkflowKit | 1,083 | Shortcuts/automation engine |
