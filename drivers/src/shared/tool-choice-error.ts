import { LlumiverseError, type LlumiverseErrorContext } from '@llumiverse/core';

export function createToolChoiceConfigurationError(message: string, context: LlumiverseErrorContext): LlumiverseError {
    const cause = new Error(message);
    cause.name = 'ToolChoiceConfigurationError';
    return new LlumiverseError(message, false, context, cause, 400, cause.name);
}
