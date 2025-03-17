# Llumiverse - Universal LLM Connectors for Node.js

![Build](https://github.com/vertesia/llumiverse/actions/workflows/build-and-test.yml/badge.svg)
[![npm version](https://badge.fury.io/js/%40llumiverse%2Fcore.svg)](https://badge.fury.io/js/%40llumiverse%2Fcore)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

LLumiverse is a universal interface for interacting with Large Language Models, for the Typescript/Javascript ecosystem. It provides a lightweight modular library for interacting with various LLM models and execution platforms.

It solely focuses on abstracting LLMs and their execution platforms, and does not provide prompt templating, or RAG, or chains, letting you pick the best tool for the job.

The following LLM platforms are supported in the current version:

| Provider | Completion | Chat | Model Listing | Multimodal | Fine-Tuning | Batch Inference |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| AWS Bedrock | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Azure OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Google Vertex AI | ✅ | ✅ | ✅ | ✅ | By Request | ✅ |
| Groq | ✅ | ✅ | ✅ | N/A | N/A | N/A |
| HuggingFace Inference Endpoints | ✅ | ✅ | N/A | N/A | N/A | N/A |
| IBM WatsonX | ✅ | ✅ | ✅ | N/A | By Request | N/A |
| Mistral AI | ✅ | ✅ | ✅ | N/A | By Request | N/A |
| OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Replicate | ✅ | ✅ | ✅ | N/A | ✅ | N/A |
| Together AI| ✅ | ✅ | ✅ | N/A | By Request | N/A |

New capabilities and platform can easily be added by creating a new driver for the platform.


## Requirements

* node v18+, or bun 1.0+

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

### Generate embeddings

LLumiverse drivers expose a method to generate vector embeddings for a given text.
Drivers supporting embeddings as of v0.10.0 are `bedrock`, `openai`, `vertexai`.
If embeddings are not yet supported the generateEmbeddings method will throws an error.

Here is an example on using the `vertexai` driver. For the example to work you need to define a `GOOGLE_APPLICATION_CREDENTIALS` env variable to be able to access your gcloud project

```javascript
import { VertexAIDriver } from "@llumiverse/drivers";

const driver = new VertexAIDriver({
    project: 'your-project-id',
    region: 'us-central1' // your zone
});

const r = await driver.generateEmbeddings({ content: "Hello world!" });

// print the vector
console.log('Embeddings: ', r.values);
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

const r = await driver.generateEmbeddings({ 
    content: "Hello world!", 
    model: "textembedding-gecko@002",  
    task_type: "SEMANTIC_SIMILARITY"
});

// print the vector
console.log('Embeddings: ', r.values);
```

The `task_type` parameter is specific to the [textembedding-gecko model](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings). 

### Batch Inference

LLumiverse supports batch inference for OpenAI, Vertex AI and AWS Bedrock. Batch inference allows you to process multiple prompts asynchronously in a single operation, which is more efficient for large-scale inference tasks.

Each provider implements batch inference differently:
- **OpenAI**: Processes batches locally with concurrent API requests
- **Vertex AI**: Uses Google's managed batch processing service
- **AWS Bedrock**: Uses Amazon's batch inference capabilities

Here's an example of how to use batch inference with Vertex AI:

```javascript
import { Modalities, PromptRole } from '@llumiverse/core';
import { VertexAIDriver } from '@llumiverse/drivers/vertexai';

// Create a Vertex AI driver
const driver = new VertexAIDriver({
  project: 'your-gcp-project-id',
  region: 'us-central1'
});

// Prepare multiple prompts for batch processing
const batchInputs = [];

// Add several sample prompts
for (let i = 0; i < 5; i++) {
  batchInputs.push({
    segments: [
      {
        role: PromptRole.user,
        content: `Tell me an interesting fact about topic ${i+1}.`
      }
    ],
    options: {
      model: 'gemini-1.0-pro',
      output_modality: Modalities.text,
      batch_id: `item-${i}`  // Assign a unique ID to each item
    }
  });
}

// Start the batch inference job
const batchJob = await driver.startBatchInference(batchInputs, {
  // Optional batch options
  maxWorkerCount: 2,
  machineType: 'e2-standard-4'
});

console.log(`Batch job created with ID: ${batchJob.id}`);

// Poll for job status
let complete = false;
while (!complete) {
  await new Promise(resolve => setTimeout(resolve, 10000)); // Wait 10 seconds
  
  const jobStatus = await driver.getBatchInferenceJob(batchJob.id);
  console.log(`Job status: ${jobStatus.status}`);
  
  if (jobStatus.status === 'succeeded' || 
      jobStatus.status === 'failed' || 
      jobStatus.status === 'cancelled' ||
      jobStatus.status === 'partial') {
    complete = true;
  }
}

// Get results if job completed successfully
const results = await driver.getBatchInferenceResults(batchJob.id);
console.log('Batch Results:', results.results);
```

Similar functionality is available for AWS Bedrock with the `BedrockDriver`.


## Contributing

Contributions are welcome!
Please see [CONTRIBUTING.md](https://github.com/vertesia/llumiverse/blob/main/CONTRIBUTING.md) for more details.


## License

Llumiverse is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Feel free to use it accordingly.
