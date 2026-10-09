import type { Logger } from '@llumiverse/core';

/**
 * Caller options normally pass through for provider validation. Report legacy compatibility
 * exceptions only when they change supplied options; never include option values in logs.
 */
export function logModelOptionException(
    logger: Logger | undefined,
    model: string,
    options: object | undefined,
    optionNames: string[],
    reason: string,
): void {
    const option_names = optionNames.filter(
        (name) => (options as Record<string, unknown> | undefined)?.[name] !== undefined,
    );
    if (option_names.length > 0) {
        logger?.warn({ model, option_names, reason }, 'Model option compatibility exception changed caller input');
    }
}
