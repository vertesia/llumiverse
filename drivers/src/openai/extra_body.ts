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
): RequestT {
    return extraBody ? ({ ...extraBody, ...request } as RequestT) : request;
}
