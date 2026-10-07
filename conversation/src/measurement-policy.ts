import type { ContextMeasurement, ProcessingBudget } from './types.js';

/** A provider count is complete only when the host has verified the whole native request. */
export function acceptsProcessingMeasurement(
    budget: ProcessingBudget,
    measurement: ContextMeasurement,
    providerCountComplete = false,
): boolean {
    if (measurement.method === 'exact') return true;
    if (measurement.method === 'provider_counted') return providerCountComplete;
    return budget.measurement_policy === 'identified_estimate' && measurement.tokenizer_version !== undefined;
}
