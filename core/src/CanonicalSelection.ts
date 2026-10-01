import { type ExecutionOptions, LlumiverseError } from '@llumiverse/common';
import type { DecodedConversationResponse } from '@llumiverse/conversation';

export type CanonicalToolSelectionPolicy =
    | { mode: 'auto' }
    | { mode: 'none' }
    | { mode: 'required'; tool_name?: string };

export const CANONICAL_REQUIRED_TOOL_CALL_MISSING = 'CANONICAL_REQUIRED_TOOL_CALL_MISSING';
export const CANONICAL_FORBIDDEN_TOOL_CALL = 'CANONICAL_FORBIDDEN_TOOL_CALL';

export type CanonicalToolSelectionDiagnosticCode =
    | typeof CANONICAL_REQUIRED_TOOL_CALL_MISSING
    | typeof CANONICAL_FORBIDDEN_TOOL_CALL;

export interface CanonicalToolSelectionDiagnostic {
    code: CanonicalToolSelectionDiagnosticCode;
    message: string;
    retryable: false;
}

export class CanonicalToolSelectionViolationError extends LlumiverseError {
    constructor(
        decoded: DecodedConversationResponse,
        readonly diagnostic_code: CanonicalToolSelectionDiagnosticCode = CANONICAL_REQUIRED_TOOL_CALL_MISSING,
    ) {
        super(
            'Canonical response violated the requested tool-selection policy',
            false,
            {
                provider: decoded.generation.provider,
                model: decoded.generation.requested_model,
                operation: 'execute',
            },
            undefined,
            undefined,
            'CanonicalToolSelectionViolationError',
        );
    }
}

/** Return the bounded public diagnostic for a pre-ingestion selection violation. */
export function canonicalToolSelectionDiagnostic(error: unknown): CanonicalToolSelectionDiagnostic | undefined {
    if (!(error instanceof CanonicalToolSelectionViolationError)) return undefined;
    return {
        code: error.diagnostic_code,
        message:
            error.diagnostic_code === CANONICAL_REQUIRED_TOOL_CALL_MISSING
                ? 'Canonical response omitted a required tool call'
                : 'Canonical response included a forbidden tool call',
        retryable: false,
    };
}

/** Normalize the effective private model-tool selection without changing omitted provider defaults. */
export function canonicalToolSelectionPolicy(
    options: Pick<ExecutionOptions, 'model_options'>,
): CanonicalToolSelectionPolicy | undefined {
    const modelOptions = options.model_options as { required_tool_name?: unknown; tool_choice?: unknown } | undefined;
    const requiredToolName = modelOptions?.required_tool_name;
    if (requiredToolName !== undefined) {
        if (typeof requiredToolName !== 'string' || requiredToolName.length === 0) {
            throw new TypeError('required_tool_name must be a nonempty string');
        }
        return { mode: 'required', tool_name: requiredToolName };
    }
    switch (modelOptions?.tool_choice) {
        case undefined:
            return undefined;
        case 'auto':
            return { mode: 'auto' };
        case 'none':
            return { mode: 'none' };
        case 'any':
        case 'required':
            return { mode: 'required' };
        default:
            throw new TypeError('tool_choice has an unsupported value');
    }
}

/** Parse the receipt marker strictly; a present malformed marker is never treated as a legacy omission. */
export function parseCanonicalToolSelectionPolicy(value: unknown): CanonicalToolSelectionPolicy {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
        throw new TypeError('Canonical tool-selection marker is invalid');
    }
    const candidate = value as Record<string, unknown>;
    if (Object.keys(candidate).some((key) => key !== 'mode' && key !== 'tool_name')) {
        throw new TypeError('Canonical tool-selection marker contains an unknown field');
    }
    if (candidate.mode === 'auto' || candidate.mode === 'none') {
        if (candidate.tool_name !== undefined) {
            throw new TypeError('Canonical tool-selection marker has an invalid tool_name');
        }
        return { mode: candidate.mode };
    }
    if (candidate.mode !== 'required') throw new TypeError('Canonical tool-selection marker has an invalid mode');
    if (candidate.tool_name === undefined) return { mode: 'required' };
    if (typeof candidate.tool_name !== 'string' || candidate.tool_name.length === 0) {
        throw new TypeError('Canonical tool-selection marker has an invalid tool_name');
    }
    return { mode: 'required', tool_name: candidate.tool_name };
}

/** Reject a fresh decoded response before any canonical generation, turn, or operation receipt is appended. */
export function assertDecodedCanonicalToolSelection(
    decoded: DecodedConversationResponse,
    policy: CanonicalToolSelectionPolicy | undefined,
): void {
    if (policy === undefined || policy.mode === 'auto') return;
    const toolCalls = decoded.turns.flatMap((turn) => turn.blocks.filter((block) => block.type === 'tool_call'));
    if (policy.mode === 'none') {
        if (toolCalls.length > 0) {
            throw new CanonicalToolSelectionViolationError(decoded, CANONICAL_FORBIDDEN_TOOL_CALL);
        }
        return;
    }
    if (policy.tool_name === undefined) {
        if (toolCalls.length === 0) throw new CanonicalToolSelectionViolationError(decoded);
        return;
    }
    if (!toolCalls.some((block) => block.tool_name === policy.tool_name)) {
        throw new CanonicalToolSelectionViolationError(decoded);
    }
}
