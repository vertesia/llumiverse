import { CompletionStream, ExecutionOptions, ExecutionResponse, PromptSegment } from "@llumiverse/core";
import { createValidationErrorCompletion, sleep } from "./utils.js";


export class TestValidationErrorCompletionStream implements CompletionStream<PromptSegment[]> {

    completion: ExecutionResponse<PromptSegment[]> | undefined;

    constructor(public segments: PromptSegment[],
        public options: ExecutionOptions) {
    }
    async *[Symbol.asyncIterator]() {
        yield "Started TestValidationError.\n";
        await sleep(1000);
        yield "chunk1\n";
        await sleep(1000);
        yield "chunk2\n";
        await sleep(1000);
        this.completion = createValidationErrorCompletion(this.segments);
    }
}
