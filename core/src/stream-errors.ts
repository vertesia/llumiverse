import { LlumiverseError, type ToolUse } from '@llumiverse/common';

export class MalformedStreamingToolArgumentsError extends LlumiverseError {
    constructor(
        tool: ToolUse<unknown>,
        finishReason: string | undefined,
        context: { provider: string; model: string },
        originalError: unknown,
    ) {
        const argumentChars = typeof tool.tool_input === 'string' ? tool.tool_input.length : 0;
        super(
            `[${context.provider}] Received malformed JSON arguments for streamed tool "${tool.tool_name || 'unknown'}" ` +
                `(finish_reason=${finishReason ?? 'unknown'}, argument_chars=${argumentChars})`,
            false,
            { ...context, operation: 'stream' },
            originalError,
            undefined,
            'MalformedStreamingToolArgumentsError',
        );
    }
}
