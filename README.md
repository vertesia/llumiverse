# Llumiverse - The Universal, Lightweight LLM Driver

![Build](https://github.com/vertesia/llumiverse/actions/workflows/build-and-test.yml/badge.svg)
[![npm version](https://badge.fury.io/js/%40llumiverse%2Fcore.svg)](https://badge.fury.io/js/%40llumiverse%2Fcore)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**Llumiverse** is the unified connectivity layer for the Large Language Model ecosystem. It provides a robust, modular, and strongly-typed interface for interacting with almost any AI provider in Node.js, Bun, and TypeScript environments.

> **Think of it as the "JDBC" or "ODBC" for LLMs.**

It solely focuses on **abstracting the connection and execution protocol**, letting you switch providers with zero code changes. It does **not** enforce prompt templating, rigid chain structures, or "magic" orchestration logic. You build the application; we handle the plumbing.

### Why Llumiverse?

*   **🚫 No Vendor Lock-in:** Switch from OpenAI to Vertex AI to Bedrock in minutes.
*   **🧩 Modular & Lightweight:** Install only what you need. Zero bloat.
*   **🛡️ Type-Safe:** First-class TypeScript support with normalized interfaces for completion, streaming, and tool use.
*   **⚡ Universal Support:** Works in Node.js, serverless functions, and Bun.

---

### Supported Platforms

| Provider | Completion | Chat | Streaming | Multimodal | Tool Calling | Embeddings |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| **AWS Bedrock** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Azure Foundry** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Azure OpenAI** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Google Vertex AI** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Groq** | ✅ | ✅ | ✅ | N/A | ✅ | N/A |
| **HuggingFace** | ✅ | ✅ | ✅ | N/A | N/A | N/A |
| **IBM WatsonX** | ✅ | ✅ | ✅ | N/A | N/A | ✅ |
| **Mistral AI** | ✅ | ✅ | ✅ | N/A | ✅ | ✅ |
| **OpenAI** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **OpenAI Compatible** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Replicate** | ✅ | ✅ | ✅ | N/A | N/A | N/A |
| **Together AI** | ✅ | ✅ | ✅ | N/A | N/A | N/A |
| **xAI (Grok)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

### Llumiverse vs. The Rest

**Vs. LangChain / LlamaIndex:**
These are **application frameworks** that provide high-level abstractions for chains, memory, and RAG. Llumiverse is a **driver** library. It operates at a lower level.
*   *Use Llumiverse if:* You want full control over your prompt logic, you want to build your own custom framework, or you simply need a clean, unified way to call an API without the weight of a massive dependency tree.
*   *Use LangChain if:* You want pre-built "chains" and are okay with higher complexity and abstraction overhead.

**Vs. Vercel AI SDK:**
Vercel's SDK is fantastic for frontend/full-stack React/Next.js applications.
*   *Use Llumiverse if:* You are building backend services, background workers, CLI tools, or generic Node.js applications where frontend-specific hooks aren't relevant. Llumiverse is environment-agnostic.

---

## Installation

**For most use cases (recommended):**
```bash
npm install @llumiverse/core @llumiverse/drivers
```

*   `@llumiverse/core`: The interfaces, types, and base abstractions.
*   `@llumiverse/drivers`: The concrete implementations for all supported providers.

## Quick Start

### 1. Initialize a Driver

Each driver takes normalized options but may require specific credentials (API keys, regions, project IDs).

```typescript
// OpenAI
import { OpenAIDriver } from "@llumiverse/drivers";
const driver = new OpenAIDriver({ apiKey: process.env.OPENAI_API_KEY });

// Google Vertex AI
import { VertexAIDriver } from "@llumiverse/drivers";
const driver = new VertexAIDriver({
    project: process.env.GOOGLE_PROJECT_ID,
    region: 'us-central1'
});

// AWS Bedrock (uses local AWS credentials automatically)
import { BedrockDriver } from "@llumiverse/drivers";
const driver = new BedrockDriver({ region: 'us-west-2' });

// ... and many more!
```

### 2. Execute a Prompt (Standard)

The execution API is normalized. You send a `PromptSegment[]` and get a `Completion` object, regardless of the provider.

```typescript
import { PromptRole } from "@llumiverse/core";

// 1. Create a generic prompt
const prompt = [
    {
        role: PromptRole.system,
        content: "You are a helpful coding assistant."
    },
    {
        role: PromptRole.user,
        content: "Write a hello world in Python."
    }
];

// 2. Execute against ANY driver
const response = await driver.execute(prompt, {
    model: "gpt-4o", // or "gemini-1.5-pro", "anthropic.claude-3-sonnet-20240229-v1:0", etc.
    temperature: 0.7,
    max_tokens: 1024
});

console.log(response.result[0].value);
```

### 3. Stream a Response

Streaming is standardized into an `AsyncIterable`.

```typescript
const stream = await driver.stream(prompt, {
    model: "gpt-4o",
    temperature: 0.7
});

for await (const chunk of stream) {
    process.stdout.write(chunk);
}

// Access the full assembled response afterwards if needed
console.log("\nFull response:", stream.completion.result[0].value);
```

### 4. Generate Embeddings

Embeddings are normalized to a simple `{ values: number[] }` structure.

```typescript
const embedding = await driver.generateEmbeddings({
    content: "Llumiverse is awesome."
});

console.log(embedding.values); // [0.012, -0.34, ...]
```

## Contributing

We welcome contributions! Whether it's a new driver, a bug fix, or a docs improvement.
Please see [CONTRIBUTING.md](https://github.com/vertesia/llumiverse/blob/main/CONTRIBUTING.md) for details.

## License

Apache 2.0