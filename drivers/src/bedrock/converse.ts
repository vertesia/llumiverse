import { JSONSchema4 } from "json-schema";
import { PromptSegment, PromptRole, readStreamAsBase64 } from "@llumiverse/core";
import { ConversationRole, ConverseRequest, Message, SystemContentBlock } from "@aws-sdk/client-bedrock-runtime";

function getJSONSafetyNotice(schema: JSONSchema4) {
    return "The answer must be a JSON object using the following JSON Schema:\n" + JSON.stringify(schema, undefined, 2);
}

function roleConversion(role: PromptRole): ConversationRole {
    return role === PromptRole.assistant ? ConversationRole.ASSISTANT : ConversationRole.USER;
}

export async function fortmatConversePrompt(segments: PromptSegment[], schema?: JSONSchema4): Promise<ConverseRequest> {
    //Non-const for concat
    let system: SystemContentBlock[] = [];
    const safety: SystemContentBlock[] = [];
    let messages: Message[] = [];

    //TODO type: 'image' -> detect from f.mime_type
    for (const segment of segments) {

        const parts: Message[] = [];
        if (segment.files) for (const f of segment.files) {
            const source = await f.getStream();
            const data = await readStreamAsBase64(source);
            parts.push({
                content: [{text: data}],
                role: roleConversion(segment.role),
            })
        }

        if (segment.content) {
            parts.push({
                content: [{text: segment.content}],
                role: roleConversion(segment.role),
            })
        }

        if (segment.role === PromptRole.system) {
            system.push({text: segment.content});
        } else if (segment.role === PromptRole.safety) {
            safety.push({text: segment.content});
        } else {
            messages = messages.concat(parts);
        }
    }

    if (schema) {
        safety.push(
            {
                text: "IMPORTANT: " + getJSONSafetyNotice(schema)
            });
        system = system.concat(safety);
    }

    console.log(JSON.stringify({
        modelId: "", //placeholder value
        messages: messages,
        system: system
    }));

    return {
        modelId: "", //placeholder value
        messages: messages,
        system: system,
    }
}
