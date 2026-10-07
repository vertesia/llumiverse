import { ConversationValidationError } from './diagnostics.js';
import { fingerprintJson } from './identity.js';
import { preflightJsonInput } from './json-preflight.js';
import type { ContextEntry, ConversationContext, ConversationDocument, ConversationEditRecordRef } from './types.js';

export function preflight(input: unknown): void {
    const result = preflightJsonInput(input);
    if (!result.success)
        throw new ConversationValidationError('Conversation edit failed JSON preflight', result.diagnostics);
}
export async function recordRefs(records: readonly { id: string }[]): Promise<ConversationEditRecordRef[]> {
    const result: ConversationEditRecordRef[] = [];
    for (const record of records) result.push({ id: record.id, fingerprint: await fingerprintJson(record) });
    return result;
}
export function nextRevision(revision: number): number {
    if (!Number.isSafeInteger(revision + 1))
        throw new RangeError('Conversation edit revision exceeds safe integer range');
    return revision + 1;
}
export function cacheAfterEdit(document: ConversationDocument, entries: ContextEntry[], contentChangedAt?: number) {
    return cacheAfterContextEdit(document.context, entries, contentChangedAt);
}

/** Same edit semantics over explicit context dependencies, without a fabricated source document. */
export function cacheAfterContextEdit(
    context: ConversationContext,
    entries: ContextEntry[],
    contentChangedAt?: number,
) {
    const cache = context.cache_intent;
    if (!cache || cache.stable_through_entry_id === undefined) return cache;
    const oldIndex = context.entries.findIndex((entry) => entry.id === cache.stable_through_entry_id);
    const missing = !entries.some((entry) => entry.id === cache.stable_through_entry_id);
    const affected = missing || (contentChangedAt !== undefined && contentChangedAt <= oldIndex);
    if (!affected) return cache;
    if (cache.mode === 'required') throw new Error('Edit would invalidate required cache prefix');
    const { stable_through_entry_id: _boundary, ...retained } = cache;
    return retained;
}

/** Same invalidation as an accepted context-change exclusion; it is not a cache-preserving edit. */
export function cacheAfterContextRemoval(context: ConversationContext, removedEntryIds: ReadonlySet<string>) {
    const cache = context.cache_intent;
    if (removedEntryIds.size === 0 || !cache) return cache;
    if (cache.mode === 'required') throw new Error('Context change would invalidate required cache intent');
    if (
        cache.mode === 'auto' ||
        (cache.mode === 'off' &&
            cache.stable_through_entry_id !== undefined &&
            removedEntryIds.has(cache.stable_through_entry_id))
    ) {
        const { stable_through_entry_id: _boundary, ...retained } = cache;
        return retained;
    }
    return cache;
}
