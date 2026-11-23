import type { Logger } from '@llumiverse/common';

/**
 * The longest tool name OpenAI, Bedrock Converse, Mistral and Gemini accept.
 */
export const PROVIDER_TOOL_NAME_MAX_LENGTH = 64;

const REJECTED_TOOL_NAME_CHARS = /[^a-zA-Z0-9_-]/g;

/**
 * The tool name to send to a provider for a tool call replayed from conversation history.
 *
 * A model can emit a tool call whose name the provider later refuses to read back: an empty string,
 * `multi_tool_use.parallel`, `functions.x`, a dotted MCP-style name. The caller answers such a call with a
 * recoverable "tool not found" result, but the call itself stays in the history and the next request fails
 * with a 400 on every later turn.
 *
 * A name the request declares is returned unchanged. Any other name has every character outside
 * `[a-zA-Z0-9_-]` replaced with `_` and is cut to `maxLength`; an empty name becomes `_`. The mapping is
 * deterministic, so a call and its result rewritten separately still carry the same name.
 *
 * Use it only on the outgoing request: the stored conversation and the tool calls returned to callers keep
 * the name the model emitted.
 */
export function providerSafeToolName(
    name: string | null | undefined,
    declaredNames: ReadonlySet<string>,
    maxLength: number = PROVIDER_TOOL_NAME_MAX_LENGTH,
): string {
    const raw = name ?? '';
    if (raw && declaredNames.has(raw)) {
        return raw;
    }
    return raw.replace(REJECTED_TOOL_NAME_CHARS, '_').slice(0, maxLength) || '_';
}

/** The set of tool names a request declares, for {@link providerSafeToolName}. */
export function declaredToolNames(tools: readonly { name: string }[] | null | undefined): ReadonlySet<string> {
    return new Set(tools?.map((tool) => tool.name));
}

/**
 * Rewrites the tool names of one outgoing request with {@link providerSafeToolName} and records each rewrite,
 * so the driver can log that it changed the replayed history.
 */
export class ReplayedToolNames {
    private readonly declared: ReadonlySet<string>;
    private readonly rewrites = new Map<string, string>();

    constructor(
        tools: readonly { name: string }[] | null | undefined,
        private readonly maxLength: number = PROVIDER_TOOL_NAME_MAX_LENGTH,
    ) {
        this.declared = declaredToolNames(tools);
    }

    /** The name to send for `name`; the same input always gives the same output. */
    safe(name: string | null | undefined): string {
        const raw = name ?? '';
        const safe = providerSafeToolName(raw, this.declared, this.maxLength);
        if (safe !== raw) {
            this.rewrites.set(raw, safe);
        }
        return safe;
    }

    /** Log the rewrites made so far, if any. Tool names only: no arguments or results. */
    logRewrites(logger: Logger | undefined, context: { provider: string; model: string }): void {
        if (this.rewrites.size === 0) {
            return;
        }
        logger?.info(
            {
                ...context,
                tool_name_rewrites: [...this.rewrites].map(([from, to]) => ({ from, to })),
            },
            'Rewrote undeclared tool names the provider would reject in replayed tool calls',
        );
    }
}
