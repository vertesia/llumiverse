import { isDeepStrictEqual } from 'node:util';
import type { Logger } from '@llumiverse/core';
import { logModelOptionException } from '../shared/model-option-exceptions.js';

export type OpenAIExtraBody = Record<string, unknown>;

export function getOpenAIExtraBody(options: unknown): OpenAIExtraBody | undefined {
    if (typeof options !== 'object' || options === null || !('extra_body' in options)) return undefined;
    const extraBody = options.extra_body;
    return typeof extraBody === 'object' && extraBody !== null && !Array.isArray(extraBody)
        ? (extraBody as OpenAIExtraBody)
        : undefined;
}

/** Merge provider extensions below Llumiverse-owned fields so extensions cannot replace the request contract. */
export function mergeOpenAIExtraBody<RequestT extends object>(
    request: RequestT,
    extraBody: OpenAIExtraBody | undefined,
    logger?: Logger,
    model?: string,
): RequestT {
    if (!extraBody) return request;
    // Undefined adapter fields represent omitted options, so they must not erase caller extensions.
    const suppliedRequest = Object.fromEntries(Object.entries(request).filter(([, value]) => value !== undefined));
    // Compatibility exception: transport-owned fields take precedence over extra_body extensions.
    const overridden = Object.keys(extraBody).filter(
        (key) => Object.hasOwn(suppliedRequest, key) && !isDeepStrictEqual(extraBody[key], suppliedRequest[key]),
    );
    logModelOptionException(logger, model ?? '', extraBody, overridden, 'openai_extra_body_precedence');
    return { ...extraBody, ...suppliedRequest } as RequestT;
}
