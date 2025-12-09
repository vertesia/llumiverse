import { ExecutionResponse, PromptSegment } from "@llumiverse/core";

export function throwError(message: string, prompt: PromptSegment[]): never {
    const err = new Error(message);
    (err as any).prompt = prompt;
    throw err;
}

export function createValidationErrorCompletion(segments: PromptSegment[]) {
    return {
        result: [{ type: "text", value: "An invalid result" }],
        prompt: segments,
        execution_time: 3000,
        error: {
            code: "validation_error",
            message: "Result cannot be validated!",
        },
        token_usage: {
            result: 10,
            prompt: 10,
            total: 20,
        },
    } as ExecutionResponse<PromptSegment[]>;
}

export function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}