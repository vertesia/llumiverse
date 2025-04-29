import { supportsToolUseBedrock } from "./tools/bedrock.js";

//Returns true if the model supports tool use, false if it doesn't, and undefined if unknown
export function supportsToolUse(provider: string, model?: string, streaming?: boolean): boolean | undefined {
    switch (provider) {
        case "bedrock":
            return supportsToolUseBedrock(model ?? "", streaming);
        default:
            //TODO: Implement a loop through all providers 
            return undefined;
    }
}