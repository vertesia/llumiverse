import { preflightJsonInput } from './json-preflight.js';
import { GenerationUsageSchema } from './schemas/execution.js';
import { validateUsage } from './semantic-validation.js';
import type { GenerationUsage, SemanticConversationDiagnosticCode } from './types.js';

/** Normalized shared accounting facts, without provider reporting payloads. */
export type GenerationAccountingUsage = Omit<GenerationUsage, 'reported_usage'>;
export type GenerationUsageAccountingInspection =
    | { status: 'absent' }
    | { status: 'unmeasured'; reason: 'no_normalized_usage' }
    | { status: 'valid'; usage: GenerationAccountingUsage }
    | {
          status: 'invalid';
          reason: 'malformed' | 'resource_limit' | 'inconsistent';
          diagnostics: SemanticConversationDiagnosticCode[];
      };

const accountingSchema = GenerationUsageSchema.omit({ reported_usage: true });

/**
 * Inspect only schema-owned scalar accounting fields. Provider reporting payloads are never
 * visited, cloned, or normalized here; protocol-specific price breakdowns remain adapter facts.
 * Partial measurements stay partial. Complete-partition consistency reuses the shared validator.
 */
export function inspectGenerationUsageAccounting(input: unknown): GenerationUsageAccountingInspection {
    if (input === undefined) return { status: 'absent' };
    if (input === null || typeof input !== 'object' || Array.isArray(input))
        return { status: 'invalid', reason: 'malformed', diagnostics: [] };
    const selected: Record<string, unknown> = {};
    for (const key of Object.keys(accountingSchema.shape)) {
        const descriptor = Object.getOwnPropertyDescriptor(input, key);
        if (descriptor === undefined) continue;
        if (!Object.hasOwn(descriptor, 'value')) return { status: 'invalid', reason: 'malformed', diagnostics: [] };
        if (descriptor.value !== undefined) selected[key] = descriptor.value;
    }
    const preflight = preflightJsonInput(selected, { max_bytes: 64 * 1024, max_depth: 8, max_nodes: 512 });
    if (!preflight.success)
        return {
            status: 'invalid',
            reason: preflight.diagnostics.some((diagnostic) => diagnostic.code.startsWith('JSON_MAX_'))
                ? 'resource_limit'
                : 'malformed',
            diagnostics: [],
        };
    const parsed = accountingSchema.safeParse(selected);
    if (!parsed.success) return { status: 'invalid', reason: 'malformed', diagnostics: [] };
    if (Object.keys(parsed.data).length === 0) return { status: 'unmeasured', reason: 'no_normalized_usage' };
    const diagnostics: SemanticConversationDiagnosticCode[] = [];
    validateUsage(parsed.data, '/usage', (code) => diagnostics.push(code), 'accounting');
    if (diagnostics.length) return { status: 'invalid', reason: 'inconsistent', diagnostics };
    return { status: 'valid', usage: parsed.data };
}
