# Using model tools with Llumiverse

Tool use (aka function calling) is supported on multiple providers and models. Support and behavior vary by provider/model:
- OpenAI: supported on modern chat models (including GPT‑4.x/4o families)
- Google Vertex AI: supported on Gemini and Claude‑on‑Vertex models
- AWS Bedrock: supported on models that implement tool use via the Converse API (e.g., Claude, Nova where applicable)
- Azure AI Foundry: supported when the underlying deployment/model exposes tool calls

Notes:
- Streaming tool use is not universally supported across providers. Prefer non‑streaming when you expect tool calls.
- Use `getModelCapabilities(model, provider)` if you need to gate tool use dynamically.

## Introduction

To declare available tools, pass the `tools` property on `ExecutionOptions`. The `tools` property is an array of `ToolDefinition` objects.

```ts
export interface ToolDefinition {
  name: string,
  description?: string,
  input_schema: {
    type: 'object';
    properties?: JSONSchema | null | undefined;
    [k: string]: unknown;
  },
}
```

When the target model needs a tool output, it will respond with `finish_reason: 'tool_use'` and include a `tool_use` array in the result:

```ts
export interface ToolUse<ParamsT = JSONObject> {
  id: string,
  tool_name: string,
  tool_input: ParamsT | null
}
```

To continue, execute the requested tool(s), then send the tool results back using a `PromptSegment` with role `tool` and the corresponding `tool_use_id`.

```ts
const r = await driver.execute([
  {
    role: PromptRole.tool,
    tool_use_id: "<the id from ToolUse>",
    content: "the tool result as a string"
  },
  // my second tool execution result if any
], options);
```

## Restoring the conversation context

At each execution, restore the conversation context so the model can continue the previous response that was paused while awaiting tool output.

To restore the conversation, pass `ExecutionResponse.conversation` from the last call into `ExecutionOptions.conversation` on the next call.
Each time an execution completes the driver updates the context and returns it via `ExecutionResponse.conversation`.

**Example:**

```ts
let r = await driver.execute(myFirstPrompt, {
  // define the available tools
  tools: [
    {
      name: "myTool",
      description: "describe what the tool is useful for",
      input_schema: {
        type: "object", properties: {}
      }
    }
  ]
});

if (r.tool_use) {
  const toolMessages = r.tool_use.map((toolUse) => {
    const result = executeTool(toolUse); // your function that runs the tool
    return {
      role: PromptRole.tool,
      tool_use_id: toolUse.id,
      content: result
    };
  });
  // continue the conversation
  r = await driver.execute(toolMessages, { conversation: r.conversation });
  console.log("Response:", r.result);
}
```

## Tool function result

Send the tool function result back via a `PromptSegment` as follows:

```ts
{
  role: "tool",
  tool_use_id: "<the id>",
  content: "the result is passed here"
}
```

This sends a string to the model. Some providers (e.g., Gemini on Vertex) can accept structured tool results; to keep consistent across providers, stringify your JSON tool outputs. Drivers that support structured tool results will parse when possible; otherwise the model receives the string representation.

### Provider notes
- OpenAI: tool use + structured output supported on most modern chat models. Streaming tool use is not supported.
- Vertex AI: Gemini and Claude support tool use; provide JSON tool schemas for best results.
- Bedrock: tool use supported on models that expose tools in the Converse API (e.g., Claude). Behavior may differ by model family.
- Azure AI Foundry: support depends on the underlying deployment/model capabilities.

