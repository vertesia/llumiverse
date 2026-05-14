import { NovaMessagesPrompt } from "@llumiverse/core/formatters";

//Docs at: https://docs.aws.amazon.com/nova/latest/userguide/complete-request-schema.html
export interface NovaPayload extends NovaMessagesPrompt {
    schemaVersion: string,
    inferenceConfig?: {
        max_new_tokens?: number,
        temperature?: number,
        top_p?: number,
        top_k?: number,
        stopSequences?: [string]
    }
}