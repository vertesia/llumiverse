import type { Logger } from '@llumiverse/common';

/**
 * The longest tool name OpenAI, Bedrock Converse, Mistral and Gemini accept.
 */
export const PROVIDER_TOOL_NAME_MAX_LENGTH = 64;

/** What a provider accepts as a tool name, for {@link providerSafeToolName}. */
export interface ToolNameRules {
    /** Matches every character the provider rejects; must be a global regex. */
    rejectedChars: RegExp;
    /** Matches the characters a name may start with, when the provider restricts them. */
    validStart?: RegExp;
    maxLength: number;
}

/** OpenAI (Responses and Chat Completions), Bedrock Converse and Mistral: `[a-zA-Z0-9_-]{1,64}`. */
export const DEFAULT_TOOL_NAME_RULES: ToolNameRules = {
    rejectedChars: /[^a-zA-Z0-9_-]/g,
    maxLength: PROVIDER_TOOL_NAME_MAX_LENGTH,
};

/**
 * Gemini: letters, digits, underscores, dots, colons and dashes, starting with a letter or an underscore,
 * at most 64 characters.
 */
export const GEMINI_TOOL_NAME_RULES: ToolNameRules = {
    rejectedChars: /[^a-zA-Z0-9_.:-]/g,
    validStart: /^[a-zA-Z_]/,
    maxLength: PROVIDER_TOOL_NAME_MAX_LENGTH,
};

/**
 * The tool name to send to a provider for a tool call replayed from conversation history.
 *
 * A model can emit a tool call whose name the provider later refuses to read back: an empty string,
 * `multi_tool_use.parallel`, `functions.x`, a dotted MCP-style name. The caller answers such a call with a
 * recoverable "tool not found" result, but the call itself stays in the history and the next request fails
 * with a 400 on every later turn.
 *
 * A name the request declares is returned unchanged. Any other name has every character the provider rejects
 * replaced with `_`, gets a leading `_` when it does not start with a character the provider allows there, and
 * is cut to the provider's limit; an empty name becomes `_`. A name the provider already accepts is returned
 * unchanged, so a history that works is sent byte for byte as before. The mapping is deterministic, so a call
 * and its result rewritten separately still carry the same name.
 *
 * Use it only on the outgoing request: the stored conversation and the tool calls returned to callers keep
 * the name the model emitted.
 */
export function providerSafeToolName(
    name: string | null | undefined,
    declaredNames: ReadonlySet<string>,
    rules: ToolNameRules = DEFAULT_TOOL_NAME_RULES,
): string {
    const raw = name ?? '';
    if (raw && declaredNames.has(raw)) {
        return raw;
    }
    let safe = raw.replace(rules.rejectedChars, '_');
    if (safe && rules.validStart && !rules.validStart.test(safe)) {
        safe = `_${safe}`;
    }
    return safe.slice(0, rules.maxLength) || '_';
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
        private readonly rules: ToolNameRules = DEFAULT_TOOL_NAME_RULES,
    ) {
        this.declared = declaredToolNames(tools);
    }

    /** The name to send for `name`; the same input always gives the same output. */
    safe(name: string | null | undefined): string {
        const raw = name ?? '';
        const safe = providerSafeToolName(raw, this.declared, this.rules);
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
