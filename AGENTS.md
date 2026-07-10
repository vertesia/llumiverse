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
- **Run all tests**: `pnpm -r test`
- **Run specific tests**: `cd drivers && pnpm test -- -t "pattern"`
- **Run linting**: `pnpm lint`
- **Format files**: `pnpm format`
- **Check formatting**: `pnpm format:check`

### Verification Scope

Prefer sparse, change-scoped verification in llumiverse unless the user explicitly asks for a full provider sweep. Many driver tests make live network calls and can produce heavy output or fail on unrelated local credentials.

- For common package changes, run `cd common && pnpm lint`, `pnpm typecheck:test`, and the specific Vitest files that cover the edit.
- For driver changes, run `cd drivers && pnpm lint`, `pnpm typecheck:test`, and targeted tests for the touched driver or behavior.
- For shared package changes that drivers consume, build dependencies sequentially, for example `common`, then `core`, then `drivers`; do not run these builds in parallel because each package build clears its own `lib` output.
- Run full `pnpm test` / `pnpm -r test` only when the change affects broad cross-provider behavior or when requested. If skipped, report the scoped commands that were run.

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
- `drivers/test/all-models.test.ts` - Tests for all models across providers
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
