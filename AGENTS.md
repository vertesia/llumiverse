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

### Finish Reasons

Drivers must return the shared finish reasons `stop`, `length`, and `tool_use` in lowercase, mapping each
provider's equivalent value to the shared spelling in both streaming and non-streaming completions. Other finish
reasons may retain the provider's native spelling and format. Keep provider-specific reasons intact so callers can
still inspect them.

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

Prefer sparse, change-scoped verification in llumiverse unless the user explicitly asks for a full provider sweep. Live tests call provider APIs and can produce heavy output or fail on unrelated local credentials.

- For common package changes, run `cd common && pnpm lint`, `pnpm typecheck:test`, and the specific functional Vitest files that cover the edit.
- For driver changes, run `cd drivers && pnpm lint`, `pnpm typecheck:test`, and targeted functional tests for the touched driver or behavior.
- Run `pnpm build` from the Llumiverse root for the normal workspace build; it uses Turbo to order package dependencies.
- Use `pnpm clean:outputs && pnpm build` only when diagnosing stale generated artifacts or a failed build after branch changes.
- Files named `*.live.test.ts` call provider APIs and may consume quota or tokens; run only the affected provider/model suite, while running functional tests freely. Only live files in the allow list in `.github/workflows/build-and-test.yml` run in CI; see "Testing" in `README.md` for the three tiers.
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

## External Input and Provider Defaults

External input means anything supplied to llumiverse by its caller, including values supplied by an application or UI.

- Preserve supplied values. Do not silently drop, clamp, replace, or otherwise repair an option merely because
  llumiverse's metadata says it is unsupported or outside a known limit. When llumiverse can construct and transport
  the request, let the provider validate it and return a clear error.
- Preserve omission. Leave omitted options undefined and omit them from the provider request wherever possible,
  so provider defaults apply. Do not fill them from option metadata, model limits, recommended values, or copied
  provider defaults. This lets provider defaults evolve without requiring a llumiverse release.
- Option and capability metadata describe known support; they are not runtime allowlists. An option's metadata
  default must not become an implicitly supplied value during execution. Use defaults in metadata only when they
  accurately describe an intentional llumiverse default, not to prescribe a caller's choice.
- Translating llumiverse options into provider field names and formats is expected; preserve their meaning and the
  distinction between an omitted value and an explicit value such as `0` or `false`.
- New exceptions must be narrow and necessary, such as a required provider field or a model workaround.
  Label each exception in a nearby code comment, explain why it is needed and when it applies, and test its scope.
  A default added for omission must not overwrite a supplied value unless that overwrite is itself an explicit exception.
- Preserve existing exceptions unless changing or removing them has been discussed with the user. These rules alone
  are not authorization to remove established compatibility behavior.
- When an exception removes or changes supplied input, log which options were affected and why through the existing
  logger. Keep logs free of prompts, credentials, and other sensitive content. Do not make input changes opaque.

## Model Compatibility and Forward Compatibility

Provider model catalogs evolve independently of llumiverse releases. Model support must therefore be expressed through
stable routing rules rather than duplicated catalog snapshots. A model should not require an explicit catalog entry
or a llumiverse release to have a reasonable chance of working on its first day of provider availability.

- Classify models by provider, publisher, model family, and generation markers. Do not use an exhaustive list of model
  IDs as a production allowlist when a family-level rule can describe the behavior.
- A newly discovered model in a known family should inherit the protocol, options, capabilities, and execution behavior
  of the newest understood generation in that family **on that provider**, while preserving provider defaults and
  external input as described above. For version-dependent behavior, use parsed versions and
  greater-than-or-equal comparisons so later generations receive the latest behavior by default.
- The same model can have different options, limits, capabilities, and transports on different providers. Resolve
  support in the target provider's context; do not assume direct-provider support or limits apply to a hosted version.
- Keep exact-ID exceptions only for documented model-specific behavior that cannot be represented by a family or
  version rule. Explain the exception next to the code.
- For regional model listings, show one regional variant per model: prefer the parent driver's configured region
  when available; otherwise use an available US region. Always retain a global listing when available, alongside
  the selected regional variant. Do not list every regional duplicate. This listing policy does not rewrite explicit
  caller-supplied model locations during execution.
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
- **Bedrock (Converse)**: Stores images as `Uint8Array` which corrupts during `JSON.stringify()` (becomes `{ "0": 137, "1": 80, ... }`). This covers non-Claude models and Claude conversations started on Converse; Claude on Bedrock now uses the Anthropic Messages API and stores base64 like the other Claude providers.
- **OpenAI/Gemini/Claude**: Stores images as base64 strings which can bloat storage

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
- Bedrock (Converse): `Uint8Array` is converted to `{ _base64: '...' }` for safe JSON serialization, then deserialized back before API calls
- OpenAI/Gemini/Claude: Base64 strings are preserved as-is (already JSON-safe)

Turn counting is automatic via `_llumiverse_meta.turnNumber` in the conversation object.

### Claude Tool Changes

Claude conversations keep `tools` append-only: tools offered after the first request are sent with
`defer_loading: true`, and additions and removals are announced by `tool_addition` / `tool_removal` blocks in
`role: "system"` messages (beta `mid-conversation-tool-changes-2026-07-01`). The declared tools are stored in
`_llumiverse_meta.claude_tools`, so keep `_llumiverse_meta` with the conversation when storing or editing it.
Unsupported models and providers (Microsoft Foundry) receive `tools` as before, without the tool-change blocks.

### Utility Functions

The following functions are exported from `@llumiverse/core` for advanced use cases:

- `stripBinaryFromConversation(obj, options?)` - Strip/serialize Bedrock binary data
- `stripBase64ImagesFromConversation(obj, options?)` - Strip OpenAI/Gemini base64 images
- `deserializeBinaryFromStorage(obj)` - Restore `Uint8Array` from serialized `{ _base64: '...' }` format
- `getConversationMeta(conversation)` - Get turn metadata
- `incrementConversationTurn(conversation)` - Increment turn number

## Testing

<<<<<<< HEAD
Live tests (`*.live.test.ts`) call provider APIs; each provider's tests run only when its credentials are set in
your local environment and are skipped otherwise.

Live tests in `drivers/test/`:
- `model-smoke.live.test.ts` - Curated smoke coverage: listing, execute, streaming with a schema, max_tokens at the
  documented limit, and image input, for one or two small current models per provider
- `conversation.live.test.ts` - Multi-turn conversations mixing execute and stream, images, and Claude prompt caching
- `tools.live.test.ts` - Tool call, tool result, then a `tools=[]` checkpoint turn
- `embeddings.live.test.ts` - Embeddings with each provider's default model
- `vertexai-embedding-models.live.test.ts` - One Vertex embedding model per request path
- `api-key-list-models.live.test.ts` - Gemini API key model listing

Live tests use small, current models so a retirement does not break CI, and build options with
`getLowEffortTestOptions()` from `drivers/test/utils.ts`: low effort where supported, no `temperature`, and only the
controls the model advertises.

Functional tests run without credentials: colocated `*.test.ts` files under `common/src`, `core/src` and
`drivers/src`, plus `core/test/` and `drivers/test/` (for example `core/test/conversation-strip.test.ts` for the
conversation stripping utilities and `drivers/test/image-gen.test.ts` for the Nova Canvas payload).
=======
Tests are functional (`*.test.ts`, no provider calls), live (`*.live.test.ts`, run on demand), or live and in CI
(live files in the allow list in `.github/workflows/build-and-test.yml`). `README.md` ("Testing") documents the tiers
and the credentials each provider needs. A test that calls a provider must be in a `*.live.test.ts` file.

Live tests in CI, under `drivers/test/`:
- `model-smoke.live.test.ts` - Curated smoke coverage for a small set of models across providers
- `conversation.live.test.ts` - Multi-turn conversations, images in conversations, Claude prompt caching
- `tools.live.test.ts` - Tool call, tool result, then a summary with tools turned off
- `embeddings.live.test.ts` - Embeddings per provider

Live tests run on demand:
- `max-tokens-limit.live.test.ts` - Each smoke model at its documented output limit (`live-smoke-drivers.ts` holds the shared model list)
- `vertexai-embedding-models.live.test.ts` - Older Vertex embedding models, one per request path
- `api-key-list-models.live.test.ts` - Gemini model listing with an API key
>>>>>>> c1a61c4 (test: stop live smoke tests depending on external hosts, and allow-list live tests in CI (#791))

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

## Callable Driver Behavior

Driver refactors must preserve callable behavior inherited or exposed by existing drivers. Do not remove methods or
add provider capability guards merely because an operation is believed to be unsupported. When llumiverse can
construct and transport the request, allow the provider SDK or API to answer and normalize its error clearly; this
also supports compatible proxies, future provider rollout, and testing. Fail locally only when llumiverse cannot
construct or transport the operation, and never silently no-op.

## Examples

Examples demonstrating how to use Llumiverse can be found in the `examples/src` directory:
- `examples/src/openai.ts` - Example for using OpenAI
- `examples/src/bedrock.ts` - Example for using AWS Bedrock
- `examples/src/vertexai.ts` - Example for using Google Vertex AI
- And more for other providers

<!-- BEGIN:turborepo-agent-rules -->

# This is NOT the Turborepo you know

Turborepo configuration, task behavior, and CLI commands can vary between installed versions and may differ from your training data. Resolve the `turbo` package from this file's directory or relevant workspace; in monorepos, it may not be visible from the repository root. For example, run `node -p "require.resolve('turbo/package.json')"` from a workspace that depends on `turbo`.

Read `docs/README.md` inside that installed package first, then read the relevant pages from its `docs/` directory before changing Turborepo configuration or commands. Heed deprecation notices. These bundled docs match the installed package version and are available without network access.

This block is written and re-added by `turbo` before repository-scoped commands when an AI agent is detected. In the Turborepo source repository, its template is defined in `crates/turborepo-cli/src/cli/agent_guidance.rs`. Removing the managed block while updates are enabled means a later qualifying invocation will add it again. Set `"agentGuidance": false` in the root `turbo.json` or `turbo.jsonc` to opt out; this does not remove an existing block. Keep the block committed with your work to avoid an uncommitted change on the next agent invocation.
<!-- END:turborepo-agent-rules -->
