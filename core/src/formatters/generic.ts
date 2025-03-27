import { JSONSchema } from "../types.js";
import { PromptRole, PromptSegment } from "../types.js";
import { getJSONSafetyNotice } from "./commons.js";

interface Labels {
    user: string,
    system: string,
    assistant: string,
    safety: string,
    instruction: string
}

export function createTextPromptFormatter(labels: Labels = {
    user: "USER",
    system: "CONTEXT",
    assistant: "ASSISTANT",
    safety: "IMPORTANT",
    instruction: "INSTRUCTION"
}) {
    return function genericTextPrompt(segments: PromptSegment[], schema?: JSONSchema): string {
        const isChat = segments.find(m => m.role === PromptRole.assistant);
        const context: string[] = [];
        const content: string[] = [];
        const safety: string[] = [];
        for (const segment of segments) {
            switch (segment.role) {
                case PromptRole.user:
                    if (isChat) {
                        content.push(`${labels.user}: ${segment.content}`);
                    } else {
                        content.push(segment.content);
                    }
                    break;
                case PromptRole.assistant:
                    content.push(`${labels.assistant}: ${segment.content}`);
                    break;
                case PromptRole.system:
                    context.push(segment.content);
                    break;
                case PromptRole.safety:
                    safety.push(segment.content);
                    break;
            }
        }

        if (schema) {
            safety.push(getJSONSafetyNotice(schema));
        }

        const out = [];
        if (context.length > 0) {
            out.push(`${labels.system}: ${context.join('\n')}`);
        }
        if (content.length > 0) {
            const prefix = context.length > 0 && !isChat ? `${labels.instruction}: ` : '';
            out.push(prefix + content.join('\n'));
        }
        if (safety.length > 0) {
            out.push(`${labels.safety}: ${safety.join('\n')}`);
        }
        return out.join('\n');
    }
}

const formatTextPrompt = createTextPromptFormatter();
export { formatTextPrompt };
