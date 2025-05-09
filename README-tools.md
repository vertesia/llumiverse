# Using model tools with llumiverse

Only gemini and claude on vertexai are supporting tools for now.

## Introduction

To declare the existing tools you can use the `tools` property on the `ExecutionOptions` to pass the tools definitions. The `tools` property ios an array of `ToolDefinition` objects.

```ts
export interface ToolDefinition {
    name: string,
    description?: string,
    input_schema: {
        type: 'object';
        properties?: unknown | null | undefined;
        [k: string]: unknown;
    },
}
```

When the target model will need a tool output it will respond using a `finish_reason` with the value of `tool_use`.
Additionally, a `tool_use` property will be available on the execution result object. The property is an array of tool use definitions:

```ts
export interface ToolUse {
    id: string,
    name: string,
    input: JSONObject | null
}
```

To continue you need to execute the tool (or the tools) requested by the model and then you send the tool results to the model using a `PromptSegment` with a role of `tool`.

```js
const r = await driver.execute([
    {
        role: PromptRole.tool,
        tool_use_id: "use the corresponding id from the ToolUse",
        content: "the tool result as a string"
    },
    // my second tool execution result if any
], options);
```

## Restoring the conversation context

At each execution you need to restore the conversation context since the model need to continue generating the previous response which was postponed while awaiting for the tool output.

In order to restore the conversation you need to recover the current state of the conversation from the last execution result and pass it on the new execution.

You can get the current state of a conversation after an execution from the `conversation` property of the `ExecutionResponse` object.
Then you need to pass this object to the next execution through the `conversation` field of the `ExecutionOptions` object.
Each time an execution is completed the driver will update the conversation and returns it back through the `ExecutionResponse.conversion` property.
When you execute the first prompt you don't need to pass any conversation.

**Example:**

```js
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
        const result = executeTool(toolUse)
        return {
            role: PromptRole.tool,
            tool_use_id: toolUse.id,
            content: result
        }
    })
    // send
    r = await driver.execute(toolMessages, {conversation: r.conversation});
    console.log("Response: ", r.result)
}
```

## Tool Function result

The result of a tool function is passed through a PromptSegment as follows:

```js
{
    role: "tool",
    content: "the result is passed here"
}
```

This means the target model will receive a string.
The `gemini-pro` model accepts a JSON object as the function result. To be able to pass an object in the case of gemini-pro you need to stringify the JSON object and pass it as a string. The gemini-pro driver will parse the JSON string if a valid JSON is found and will send the object as the result to the model. If no valid JSON is found it will send the string as the result.
