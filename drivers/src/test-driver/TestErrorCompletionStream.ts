import { CompletionStream, ExecutionOptions, ExecutionResponse, PromptSegment } from "@llumiverse/core";
import { sleep, throwError } from "./utils.js";

export class TestErrorCompletionStream implements CompletionStream<PromptSegment[]> {

    completion: ExecutionResponse<PromptSegment[]> | undefined;

    constructor(public segments: PromptSegment[],
        public options: ExecutionOptions) {
    }
    async *[Symbol.asyncIterator]() {
        yield "Started TestError. Next we will thrown an error.\n";
        sleep(1000);
        throwError("Testing stream completion error.", this.segments);
    }
}

