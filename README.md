# Llumiverse - Universal LLM Connectors for Node.js

![Build](https://github.com/vertesia/llumiverse/actions/workflows/build-and-test.yml/badge.svg)
[![npm version](https://badge.fury.io/js/%40llumiverse%2Fcore.svg)](https://badge.fury.io/js/%40llumiverse%2Fcore)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

LLumiverse is a universal interface for interacting with Large Language Models, for the TypeScript/JavaScript ecosystem. It provides a lightweight, modular library for interacting with various LLM models and execution platforms.

It solely focuses on abstracting LLMs and their execution platforms, and does not provide prompt templating, or RAG, or chains, letting you pick the best tool for the job.

The following LLM platforms are supported in the current version:

| Provider | Completion | Chat | Model Listing | Multimodal | Fine-Tuning |
| --- | :-: | :-: | :-: | :-: | :-: |
| AWS Bedrock | ✅ | ✅ | ✅ | ✅ | ✅ |
| Azure OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ |
| Azure AI Foundry | ✅ | ✅ | ✅ | ✅ | By Model |
| Google Vertex AI | ✅ | ✅ | ✅ | ✅ | By Request |
| Groq | ✅ | ✅ | ✅ | N/A | N/A |
| HuggingFace Inference Endpoints | ✅ | ✅ | N/A | N/A | N/A |
| IBM WatsonX | ✅ | ✅ | ✅ | N/A | By Request |
| Mistral AI | ✅ | ✅ | ✅ | N/A | By Request |
| OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ |
| Replicate | ✅ | ✅ | ✅ | N/A | ✅ |
| Together AI| ✅ | ✅ | ✅ | N/A | By Request |

New capabilities and platform can easily be added by creating a new driver for the platform.


## Architecture

Llumiverse is a pnpm monorepo with three main packages:
- `@llumiverse/common` — shared types, enums, capability detection, and model option metadata
- `@llumiverse/core` — driver abstraction (`Driver`, `AbstractDriver`), prompt formatting, streaming, and result validation
- `@llumiverse/drivers` — provider-specific drivers (OpenAI, Bedrock, Vertex AI, Azure AI Foundry, Groq, HuggingFace IE, Mistral, Replicate, Together AI, WatsonX)

Key concepts:
- Prompts are arrays of `PromptSegment` with roles (`user`, `system`, `assistant`, `safety`, `tool`, etc.). Drivers format these per provider.
- Streaming uses a unified `CompletionStream` interface; if a provider cannot stream, a fallback streams the full response as one chunk.
- Structured output is optional: pass a JSON Schema to `result_schema`. Results are validated with Ajv and surfaced in `error` if invalid.
- Capabilities and model options can be introspected at runtime to build UIs or guard features.


## Why Llumiverse

Use Llumiverse when you want a precise, typed, provider‑agnostic LLM connector layer without bringing a full framework. It focuses on:

- Cross‑provider coverage with consistent APIs across OpenAI, Azure (OpenAI + AI Foundry), Vertex AI (Gemini/Claude/Imagen/LLama MaaS), AWS Bedrock, Groq, Mistral, Together, Replicate, HuggingFace IE, WatsonX
- Clean prompts and multimodal formatting (OpenAI‑like chat, Bedrock Nova, Imagen) plus image generation via Vertex/Bedrock
- Unified streaming with fallback, normalized finish reasons, and token usage
- Structured output via JSON Schema (Ajv validation) with provider‑level hints where supported
- Tool calling across providers with normalized `tool_use` handling and conversation carry‑over
- Embeddings with a single function for text and image (where supported)
- Model discovery, capabilities detection, and model‑specific option metadata for building UIs and guardrails
- Fine‑tuning/training surfaces (currently OpenAI) with a path to extend per provider

When not to use Llumiverse:
- You want a larger platform with orchestration, evaluation, deployment, governance, and UI tooling → consider Vertesia.
- You need a batteries‑included developer framework for chains, memory, agents, and retrieval → use a chains/agents framework.
- You need a UI‑first streaming toolkit and typed function calling geared to single‑provider apps → use a lightweight streaming SDK.

Llumiverse complements those tools by being a thin, reliable connector layer you can compose into your own architecture.


## Requirements

* node v22+, or bun 1.0+

## Installation 

1. If you want to use llumiverse to execute prompt completion on various supported providers then install `@llumiverse/core` and `@llumiverse/drivers`

```
npm install @llumiverse/core @llumiverse/drivers
```

2. If you only want to use typescript types or other structures from llumiverse you only need to install `@llumiverse/core`

```
npm install @llumiverse/core
```

3. If you want to develop a new llumiverse driver for an unsupported LLM provider you only need to install `@llumiverse/core`

```
npm install @llumiverse/core
```

## Usage

First, you need to instantiate a driver instance for the target LLM platform you want to interact too. Each driver accepts its own set of parameters when instantiating.

### OpenAI driver

```javascript
import { OpenAIDriver } from "@llumiverse/drivers";

// create an instance of the OpenAI driver 
const driver = new OpenAIDriver({
    apiKey: "YOUR_OPENAI_API_KEY"
});
```

### Azure AI Foundry driver

Use this driver when interacting with Azure AI Foundry deployments (including OpenAI-backed and other publishers). Use your project endpoint and pass the composite model id returned by `listModels()` (`deploymentName::baseModel`).

```javascript
import { AzureFoundryDriver } from "@llumiverse/drivers";

const driver = new AzureFoundryDriver({
  endpoint: process.env.AZURE_FOUNDRY_ENDPOINT,
  // Uses DefaultAzureCredential by default; override via azureADTokenProvider if needed
});

// list available deployments (as models)
const models = await driver.listModels();

// execute using the composite id returned by listModels()
const response = await driver.execute(prompt, {
  model: models[0].id, // e.g. "my-deployment::gpt-4o"
  temperature: 0.3,
  max_tokens: 512
});
```

### Bedrock driver

In this example, we will instantiate the Bedrock driver using credentials from the Shared Credentials File (i.e. ~/.aws/credentials).
Learn more on how to [setup AWS credentials in node](https://docs.aws.amazon.com/sdk-for-javascript/v3/developer-guide/setting-credentials-node.html). 

```javascript
import { BedrockDriver } from "@llumiverse/drivers";

const driver = new BedrockDriver({
    region: 'us-west-2'
});
```

### VertexAI driver

For the following example to work you need to define a `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

```javascript 
import { VertexAIDriver } from "@llumiverse/drivers";
const driver = new VertexAIDriver({
    project: 'YOUR_GCLOUD_PROJECT',
    region: 'us-central1'
});
```

### Replicate driver

```javascript
import { ReplicateDriver } from "@llumiverse/drivers";

const driver = new ReplicateDriver({
    apiKey: "YOUR_REPLICATE_API_KEY"
});
```

### TogetherAI driver

```javascript
import { TogetherAIDriver } from "@llumiverse/drivers";

const driver = new TogetherAIDriver({
    apiKey: "YOUR_TOGETHER_AI_API_KEY"
});
```

### HuggingFace driver

```javascript
import { HuggingFaceIEDriver } from "@llumiverse/drivers";

const driver = new HuggingFaceIEDriver({
    apiKey: "YOUR_HUGGINGFACE_API_KEY",
    endpoint_url: "YOUR_HUGGINGFACE_ENDPOINT",
});
```

### Listing available models

Once you instantiated a driver you can list the available models. Some drivers accept an argument for the `listModel` method to search for matching models. Some drivers like for example `replicate` are listing a preselected set of models. To list other models you need to perform a search by giving a text query as an argument.

In the following example, we are assuming that we have already instantiated a driver, which is available as the `driver` variable.


```javascript
import { AIModel } from "@llumiverse/core";

// instantiate the desired driver
const driver = createDriverInstance();

// list available models on the target LLM. (some drivers may require a search parameter to discover more models)
const models: AIModel[] = await driver.listModels();

console.log('# Available Models:');
for (const model of models) {
    console.log(`${model.name} [${model.id}]`);
}
```

### Execute a prompt 

To execute a prompt we need to create a prompt in the LLumiverse format and pass it to the driver `execute` method. 

The prompt format is very similar to the OpenAI prompt format. It is an array of messages with a `content` and a `role` property. The roles can be any of `"user" | "system" | "assistant" | "safety"`. 

The `safety` role is similar to `system` but has a greater precedence over the other messages. Thus, it will override any `user` or `system` message that is saying something contrary to the `safety` message.

In order to execute a prompt we also need to specify a target model, given a model ID which is known by the target LLM. We may also specify execution options like `temperature`, `max_tokens` etc.

In the following example, we are again assuming that we have already instantiated a driver, which is available as the `driver` variable. 

Also, we are assuming the model ID we want to target is available as the `model` variable. To get a list of the existing models (and their IDs) you can list the model as we shown in the previous example

Here is an example of model IDs depending on the driver type:
* OpenAI: `gpt-3.5-turbo`
* Bedrock: `arn:aws:bedrock:us-west-2::foundation-model/cohere.command-text-v14`
* VertexAI: `text-bison`
* Replicate: `meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3`
* TogetherAI: `mistralai/Mistral-7B-instruct-v0.1`
* HuggingFace: `aws-mistral-7b-instruct-v0-1-015`


```javascript
import { PromptRole, PromptSegment } from "@llumiverse/core";


// instantiate the desired driver
const driver = createDriverInstance();
const model = "the-model-id"; // change with your desired model ID

// create the prompt.
const prompt: PromptSegment[] = [
    {
        role: PromptRole.user,
        content: 'Please, write a short story about winter in Paris, in no more than 512 characters.'
    }
]

// execute a model and wait for the response
console.log(`\n# Executing prompt on ${model} model: ${prompt}`);
const response = await driver.execute(prompt, {
    model,
    temperature: 0.6,
    max_tokens: 1024
});

console.log('\n# LLM response:', response.result)
console.log('# Response took', response.execution_time, 'ms')
console.log('# Token usage:', response.token_usage);
```


### Execute a prompt in streaming mode

In this example, we will execute a prompt and will stream the result to display it on the console as it is returned by the target LLM platform. 

**Note** that some models don't support streaming. In that case, the driver will simulate a streaming using a single chunk of text corresponding to the entire response.


```javascript
import { PromptRole, PromptSegment } from "@llumiverse/core";

// instantiate the desired driver
const driver = createDriverInstance();
const model = "the-model-id"; // change with your desired model ID

// create the prompt.
const prompt: PromptSegment[] = [
    {
        role: PromptRole.user,
        content: 'Please, write a short story about winter in Paris, in no more than 512 characters.'
    }
]

// execute the prompt in streaming mode 
console.log(`\n# Executing prompt on model ${model} in streaming mode: ${prompt}`);
const stream = await driver.stream(prompt, {
    model,
    temperature: 0.6,
    max_tokens: 1024
});

// print the streaming response as it comes
for await (const chunk of stream) {
    process.stdout.write(chunk);
}

// when the response stream is consumed we can get the final response using stream.completion field.
const streamingResponse = stream.completion!;

console.log('\n# LLM response:', streamingResponse.result)
console.log('# Response took', streamingResponse.execution_time, 'ms')
console.log('# Token usage:', streamingResponse.token_usage);
```

### Structured output (JSON Schema)

You can request structured output by providing a JSON Schema via `result_schema`. Llumiverse will validate the final result and populate `error` if parsing/validation fails. When supported by the provider (e.g., OpenAI structured output), the driver also hints the schema to the model.

```javascript
import { PromptRole } from "@llumiverse/core";

const schema = {
  type: "object",
  properties: {
    title: { type: "string" },
    tags: { type: "array", items: { type: "string" } }
  },
  required: ["title"]
};

const prompt = [
  { role: PromptRole.user, content: "Return a blog post title and 3 tags about TypeScript." }
];

const res = await driver.execute(prompt, {
  model: "gpt-4o",
  result_schema: schema,
  temperature: 0.2,
});

if (res.error) {
  console.error("Validation error:", res.error.message);
} else {
  console.log("JSON:", res.result);
}
```

### Generate embeddings

Llumiverse drivers expose `generateEmbeddings()` to generate vector embeddings for text or images (when supported by the provider). Current drivers with embeddings support include `openai`, `vertexai` and `azure_foundry` (text and image via Inference API). Bedrock support varies by model.

Here is an example using the `vertexai` driver. For the example to work you need to define a `GOOGLE_APPLICATION_CREDENTIALS` env variable to be able to access your GCP project.

```javascript
import { VertexAIDriver } from "@llumiverse/drivers";

const driver = new VertexAIDriver({
    project: 'your-project-id',
    region: 'us-central1' // your zone
});

const r = await vertex.generateEmbeddings({ content: "Hello world!" });

// print the vector
console.log('Embeddings: ', v.values);
```

The result object contains the vector as the `values` property, the `model` used to generate the embeddings and an optional `token_count` which if defined is the token count of the input text. 
Depending on the driver, the result may contain additional properties. 

Also you can specify a specific model to be used or pass other driver supported parameter. 

**Example:**

```javascript
import { VertexAIDriver } from "@llumiverse/drivers";

const driver = new VertexAIDriver({
    project: 'your-project-id',
    region: 'us-central1' // your zone
});

const r = await vertex.generateEmbeddings({ 
    content: "Hello world!", 
    model: "textembedding-gecko@002",  
    task_type: "SEMANTIC_SIMILARITY"
});

// print the vector
console.log('Embeddings: ', v.values);
```

The `task_type` parameter is specific to the [textembedding-gecko model](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings). 

### Tool use

Some providers support model tool calling (aka “function calling”). You can pass tool definitions via `ExecutionOptions.tools` and handle tool calls by sending follow-up `PromptSegment` with role `tool`. See the dedicated guide in `README-tools.md` for end-to-end examples and provider notes.


## Contributing

Contributions are welcome!
Please see [CONTRIBUTING.md](https://github.com/vertesia/llumiverse/blob/main/CONTRIBUTING.md) for more details.


## License

Llumiverse is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Feel free to use it accordingly.
