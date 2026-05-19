# VERTESIA.md

This file provides guidance to Vertesia Code Review when working with code in this repository.

# Llumiverse

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

## Code Style

- TypeScript strict mode with noUnusedLocals/Parameters
- ESM modules with node-next resolution
- Use async/await with proper error handling (no floating promises)
- Objects: use shorthand notation
- Unused variables prefix: `_` (e.g., `_unused`)
- Line length: 120 characters, single quotes
- Component patterns: follow existing naming, directory structure and import patterns
- Always use proper typing - avoid `any` when possible
- Error handling: use proper error types and propagation, especially with async code
- Formatting: follows Prettier configuration

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

Tests can be found in:
- `drivers/test/all-models.test.ts` - Tests for all models across providers
- `drivers/test/tools.test.ts` - Tests for LLM tool functionality
- `drivers/test/embeddings.test.ts` - Tests for embedding functionality
- `drivers/test/image-gen.test.ts` - Tests for image generation

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