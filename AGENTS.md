# Llumiverse Agent Guide

Llumiverse is a universal interface for interacting with Large Language Models for the TypeScript/JavaScript ecosystem. It provides a lightweight modular library for interacting with various LLM models and execution platforms.

## Architecture

Llumiverse is structured as a monorepo with two main packages:

1. **@llumiverse/core** - Contains the core interfaces, types, and abstract classes that define the Llumiverse API.
2. **@llumiverse/drivers** - Contains implementations for various LLM providers (OpenAI, Bedrock, VertexAI, etc.).

The main abstractions are:
- `Driver` - Base interface for all LLM provider implementations
- `AbstractDriver` - Abstract class that provider-specific drivers extend
- `PromptSegment` - Represents a piece of a prompt with a role (user, system, assistant)
- `CompletionStream` - Handles streaming responses from LLMs

## Build & Test Commands

- **Build all packages**: `pnpm build`
- **Build core package**: `cd core && pnpm build`
- **Build drivers package**: `cd drivers && pnpm build`
- **Run functional tests**: `pnpm exec vitest run --exclude '**/*.live.test.ts'`
- **Run a targeted test**: `cd drivers && pnpm test -- <test-file>`
- **Run all tests, including live provider tests**: `pnpm -r test`
- **Run linting**: `pnpm lint`
- **Format files**: `pnpm format`
- **Check formatting**: `pnpm format:check`

### Verification Scope

Prefer sparse, change-scoped verification in llumiverse unless the user explicitly asks for a full provider sweep. Many driver tests make live network calls and can produce heavy output or fail on unrelated local credentials.

- For common package changes, run `cd common && pnpm lint`, `pnpm typecheck:test`, and the specific functional Vitest files that cover the edit.
- For driver changes, run `cd drivers && pnpm lint`, `pnpm typecheck:test`, and targeted functional tests for the touched driver or behavior.
- Run `pnpm build` from the Llumiverse root for the normal workspace build; it uses Turbo to order package dependencies.
- Use `pnpm clean:outputs && pnpm build` only when diagnosing stale generated artifacts or a failed build after branch changes.
- Files named `*.live.test.ts` call provider APIs and may consume quota or tokens; run only the affected provider/model suite, while running functional tests freely.
- Use `LLUMIVERSE_LIVE_PROVIDERS` and `LLUMIVERSE_LIVE_MODELS` with `drivers/test/model-smoke.live.test.ts` to select exact live smoke targets instead of relying on a Vitest name filter.
- Run the full `pnpm test` / `pnpm -r test` only when the change affects broad cross-provider behavior or when requested. If skipped, report the scoped commands that were run.

## Code Style

- TypeScript strict mode with `noUnusedLocals`/`noUnusedParameters` enabled
- ESM modules with node-next resolution
- Async patterns:
  - Use async/await with proper error handling
  - No floating promises allowed
  - Always catch and handle exceptions appropriately
- Objects: use shorthand notation where applicable
- Naming conventions:
  - Unused variables prefix: `_` (e.g., `_unused`)
  - Line length: 120 characters maximum
  - Use 4-space indentation
  - Use single quotes for strings
- Component patterns: follow existing naming, directory structure and import patterns
- Type safety:
  - Always use proper typing - avoid `any` when possible
  - Use TypeScript utility types where appropriate
- Error handling: use proper error types and propagation, especially with async code
- Formatting: follows `biome.json` in this repository

## Model Compatibility and Forward Compatibility

Provider model catalogs evolve independently of llumiverse releases. Model support must therefore be expressed through
stable routing rules rather than duplicated catalog snapshots.

- Classify models by provider, publisher, model family, and generation markers. Do not use an exhaustive list of model
  IDs as a production allowlist when a family-level rule can describe the behavior.
- A newly discovered model in a known family should inherit the protocol, options, capabilities, and execution behavior
  of the newest understood generation in that family. For version-dependent behavior, use parsed versions and
  greater-than-or-equal comparisons so later generations receive the latest behavior by default.
- Keep exact-ID exceptions only for documented model-specific behavior that cannot be represented by a family or
  version rule. Explain the exception next to the code.
- Treat the provider's model-discovery API as the source of regional and account availability. Llumiverse should filter
  or enrich those results by category; it should not maintain a second exhaustive provider catalog.
- Tests should cover category invariants, version boundaries, representative current IDs, and plausible future IDs.
  Avoid assertions over a complete ordered model list or a fixed catalog size when adding a model should remain a
  compatible change.
- Exact model inventories may appear in opt-in compatibility audits or fixtures, but they must not determine runtime
  support and should not make the normal test suite fail solely because a provider added a model.
- Prefer provider-native structured-output configuration (`response_format`, `outputConfig`, constrained output, or the
  provider equivalent) whenever the model supports it. Keep fallback behavior as explicit family-level rules for
  providers or generations with known schema-enforcement defects; do not silently make prompt alignment the default.
- Keep Claude schema behavior conservative because Claude is heavily used by customers: continue using prompt guidance
  for schemas rather than switching to strict constrained-output enforcement. Do not change this established
  prompt/schema handling unless a future Claude-specific overhaul includes dedicated regression coverage and an explicit
  compatibility review.

## Multi-Turn Conversations

Llumiverse supports multi-turn conversations via the `conversation` property in `ExecutionOptions`. The conversation object is driver-specific and opaque - you receive it from one execution and pass it to the next.

### Image Handling in Conversations

Images in conversations can cause issues:
- **Bedrock**: Stores images as `Uint8Array` which corrupts during `JSON.stringify()` (becomes `{ "0": 137, "1": 80, ... }`)
- **OpenAI/Gemini**: Stores images as base64 strings which can bloat storage

To address this, drivers automatically strip/serialize images based on `stripImagesAfterTurns`:

```typescript
// Strip images immediately after each turn (default)
driver.execute(prompt, {
    model: "...",
    stripImagesAfterTurns: 0  // or undefined
});

// Keep images for 3 turns before stripping
driver.execute(prompt, {
    model: "...",
    stripImagesAfterTurns: 3
});

// Never strip images (for short conversations)
driver.execute(prompt, {
    model: "...",
    stripImagesAfterTurns: Infinity
});
```

When `stripImagesAfterTurns > 0`:
- Bedrock: `Uint8Array` is converted to `{ _base64: '...' }` for safe JSON serialization, then deserialized back before API calls
- OpenAI/Gemini: Base64 strings are preserved as-is (already JSON-safe)

Turn counting is automatic via `_llumiverse_meta.turnNumber` in the conversation object.

### Utility Functions

The following functions are exported from `@llumiverse/core` for advanced use cases:

- `stripBinaryFromConversation(obj, options?)` - Strip/serialize Bedrock binary data
- `stripBase64ImagesFromConversation(obj, options?)` - Strip OpenAI/Gemini base64 images
- `deserializeBinaryFromStorage(obj)` - Restore `Uint8Array` from serialized `{ _base64: '...' }` format
- `getConversationMeta(conversation)` - Get turn metadata
- `incrementConversationTurn(conversation)` - Increment turn number

## Testing

Tests require API keys for the various LLM providers stored as environment variables:
- OPENAI_API_KEY
- BEDROCK_REGION
- GOOGLE_PROJECT_ID and GOOGLE_REGION
- AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT
- MISTRAL_API_KEY
- TOGETHER_API_KEY
- GROQ_API_KEY
- WATSONX_API_KEY
- XAI_API_KEY
- OPENROUTER_API_KEY

Tests can be found in:
- `drivers/test/model-smoke.live.test.ts` - Curated live smoke coverage for a small set of models across providers
- `drivers/test/tools.test.ts` - Tests for LLM tool functionality
- `drivers/test/embeddings.test.ts` - Tests for embedding functionality
- `drivers/test/image-gen.test.ts` - Tests for image generation
- `drivers/test/conversation.test.ts` - Tests for multi-turn conversations with images
- `core/test/conversation-strip.test.ts` - Unit tests for conversation stripping utilities

## Adding a New Driver

To add a new LLM provider:

1. Create a new directory in `drivers/src/` for the provider
2. Implement a driver class that extends `AbstractDriver`
3. Implement the required abstract methods
4. Add the driver to the exports in `drivers/src/index.ts`
5. Add tests in `drivers/test/`

## Adding New Features

When adding new features:
1. First add types and interfaces to `core/src/types.ts`
2. Update the `AbstractDriver` class in `core/src/Driver.ts` if needed
3. Implement the feature in the relevant drivers

## Examples

Examples demonstrating how to use Llumiverse can be found in the `examples/src` directory:
- `examples/src/openai.ts` - Example for using OpenAI
- `examples/src/bedrock.ts` - Example for using AWS Bedrock
- `examples/src/vertexai.ts` - Example for using Google Vertex AI
- And more for other providers
