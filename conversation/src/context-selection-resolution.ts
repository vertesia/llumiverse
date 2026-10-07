import { canonicalJsonContentString } from './content-integrity.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { boundConversationDiagnostic, ConversationValidationError } from './diagnostics.js';
import type {
    ContentBlock,
    ContextChangePlanInput,
    ContextEntry,
    ContextMetadataPredicate,
    ContextSelectionAnchor,
    ContextSelector,
    ConversationDocument,
    ConversationTurn,
} from './types.js';

export function rejectContextSelection(
    message: string,
    code: 'SELECTION_RANGE_INVALID' | 'CONTEXT_SELECTION_OVERLAP' | 'REFERENCE_NOT_FOUND' = 'SELECTION_RANGE_INVALID',
): never {
    throw new ConversationValidationError(message, [
        boundConversationDiagnostic({ stage: 'semantic', code, path: '/selector', message }),
    ]);
}
function anchorIndex(
    entryIndices: ReadonlyMap<string, number>,
    turnIndices: ReadonlyMap<string, readonly number[]>,
    anchor: ContextSelectionAnchor,
): number {
    const entryIndex = entryIndices.get(anchor.id);
    const matches =
        anchor.kind === 'entry' ? (entryIndex === undefined ? [] : [entryIndex]) : (turnIndices.get(anchor.id) ?? []);
    if (matches.length === 0) rejectContextSelection('Selection anchor is not active', 'REFERENCE_NOT_FOUND');
    if (matches.length !== 1) rejectContextSelection('Turn anchor is ambiguous; select an exact active entry');
    return matches[0];
}
function matchesMetadata(metadata: unknown, predicate: ContextMetadataPredicate): boolean {
    let value: unknown = metadata;
    let exists = true;
    for (const segment of predicate.path) {
        if (value === null || typeof value !== 'object' || !Object.hasOwn(value, segment)) {
            exists = false;
            break;
        }
        // The bounded JSON preflight and document parser exclude getters and inherited properties.
        value = Reflect.get(value, segment);
    }
    if (predicate.op === 'exists') return exists === predicate.exists;
    if (!exists) return false;
    const actual = canonicalJsonContentString(value);
    return predicate.op === 'equals'
        ? actual === canonicalJsonContentString(predicate.value)
        : predicate.values.some((item) => actual === canonicalJsonContentString(item));
}

/** Shared structural resolver: eligibility belongs to mutation planning, never read inspection. */
export function resolveActiveContextSelection(document: ConversationDocument, selector: ContextSelector) {
    const entries = document.context.entries;
    const turns = createContextTurnIndex(document);
    const source = selector.source;
    const selected = new Set<number>();
    const entryIndices = new Map<string, number>();
    const turnIndices = new Map<string, number[]>();
    for (const [index, entry] of entries.entries()) {
        entryIndices.set(entry.id, index);
        const indices = turnIndices.get(entry.turn_id) ?? [];
        indices.push(index);
        turnIndices.set(entry.turn_id, indices);
    }
    if (source.kind === 'all')
        entries.forEach((_, index) => {
            selected.add(index);
        });
    else if (source.kind === 'turn_ids') {
        if (new Set(source.turn_ids).size !== source.turn_ids.length)
            rejectContextSelection('Selection repeats a turn ID', 'CONTEXT_SELECTION_OVERLAP');
        for (const id of source.turn_ids) {
            const indices = turnIndices.get(id) ?? [];
            if (!indices.length) rejectContextSelection('Selected turn is not active', 'REFERENCE_NOT_FOUND');
            for (const index of indices) selected.add(index);
        }
    } else {
        const ranges = source.kind === 'range' ? [source.range] : source.ranges;
        for (const range of ranges) {
            const first = anchorIndex(entryIndices, turnIndices, range.from),
                last = anchorIndex(entryIndices, turnIndices, range.through);
            if (last < first) rejectContextSelection('Selection range is reversed');
            for (let index = first; index <= last; index += 1) {
                if (selected.has(index))
                    rejectContextSelection('Selection ranges overlap', 'CONTEXT_SELECTION_OVERLAP');
                selected.add(index);
            }
        }
    }
    const filters = selector.filters;
    const actorKinds = filters?.actor_kinds ? new Set(filters.actor_kinds) : undefined;
    const actorIds = filters?.actor_ids ? new Set(filters.actor_ids) : undefined;
    const requestedBlocks = filters?.block_ids ? new Set(filters.block_ids) : undefined;
    const blockTypes = filters?.block_types ? new Set(filters.block_types) : undefined;
    const requestedTools = filters?.tool_names ? new Set(filters.tool_names) : undefined;
    const active = entries.flatMap((entry) => resolveContextEntry(turns, entry).blocks);
    if (filters?.block_ids) {
        const activeIds = new Set(active.map((block) => block.id));
        if (new Set(filters.block_ids).size !== filters.block_ids.length)
            rejectContextSelection('Selection repeats a block ID', 'CONTEXT_SELECTION_OVERLAP');
        if (filters.block_ids.some((id) => !activeIds.has(id)))
            rejectContextSelection('Selected top-level block is not active', 'REFERENCE_NOT_FOUND');
    }
    const toolNames = new Map<string, string>();
    for (const turn of turns.values())
        for (const block of turn.blocks) if (block.type === 'tool_call') toolNames.set(block.call_id, block.tool_name);
    const matchesBlock = (block: ContentBlock): boolean => {
        if (requestedBlocks && !requestedBlocks.has(block.id)) return false;
        if (blockTypes && !blockTypes.has(block.type)) return false;
        if (requestedTools) {
            const name =
                block.type === 'tool_call'
                    ? block.tool_name
                    : block.type === 'tool_result'
                      ? toolNames.get(block.call_id)
                      : undefined;
            if (name === undefined || !requestedTools.has(name)) return false;
        }
        return true;
    };
    const matchedEntries: { entry: ContextEntry; turn: ConversationTurn; blocks: ContentBlock[] }[] = [];
    const entryIds: string[] = [];
    const selectedEntries: ContextEntry[] = [];
    const blockIds: Record<string, string[]> = {};
    let partial = false;
    for (const [index, entry] of entries.entries()) {
        if (!selected.has(index)) continue;
        const { turn, blocks } = resolveContextEntry(turns, entry);
        if (actorKinds && !actorKinds.has(turn.kind)) continue;
        if (actorIds && (turn.actor_id === undefined || !actorIds.has(turn.actor_id))) continue;
        if (filters?.metadata && !filters.metadata.every((predicate) => matchesMetadata(turn.metadata, predicate)))
            continue;
        const matched = blocks.filter(matchesBlock);
        // Empty turns can be selected by source/actor/metadata, but not a block filter.
        if (!matched.length && (blocks.length || filters?.block_ids || filters?.block_types || filters?.tool_names))
            continue;
        matchedEntries.push({ entry, turn, blocks: matched });
        entryIds.push(entry.id);
        selectedEntries.push(entry);
        if (matched.length !== blocks.length) {
            // Entry identifiers are data, including Object.prototype property names.
            Object.defineProperty(blockIds, entry.id, {
                value: matched.map((block) => block.id),
                enumerable: true,
            });
            partial = true;
        }
    }
    return { matchedEntries, entryIds, selectedEntries, blockIds, partial };
}

/** A partial selection partitions only the already-active top-level blocks, in original order. */
export function partitionContextSelection(
    document: ConversationDocument,
    input: ContextChangePlanInput,
    entries = document.context.entries,
) {
    const turns = createContextTurnIndex(document);
    const ids = new Set(input.entry_ids);
    const selected = entries.filter((entry) => ids.has(entry.id));
    if (selected.length !== input.entry_ids.length) throw new Error('Context change selects an unavailable entry');
    if (
        input.selected_entries !== undefined &&
        canonicalJsonContentString(selected) !== canonicalJsonContentString(input.selected_entries)
    ) {
        throw new Error('Partial context selection source entries conflict');
    }
    const partial = input.selected_block_ids;
    if (partial && Object.keys(partial).some((id) => !ids.has(id)))
        throw new Error('Partial selection has an unavailable entry');
    const removed: ContextEntry[] = [];
    const retained: ContextEntry[] = [];
    const segments: { entry: ContextEntry; selected: boolean; block_ids?: string[] }[] = [];
    let ranges = 0;
    let inRange = false;
    for (const entry of entries) {
        if (!ids.has(entry.id)) {
            retained.push(entry);
            segments.push({ entry, selected: false });
            inRange = false;
            continue;
        }
        const blocks = resolveContextEntry(turns, entry).blocks;
        const blockIds = partial && Object.hasOwn(partial, entry.id) ? partial[entry.id] : undefined;
        if (blockIds === undefined) {
            removed.push(entry);
            segments.push({ entry, selected: true });
            if (!inRange) ranges += 1;
            inRange = true;
            continue;
        }
        if (entry.type === 'replacement_turn' && blockIds.length !== blocks.length) {
            throw new Error('Compaction replacement is an indivisible provenance unit');
        }
        if (
            entry.type === 'source_turn' &&
            resolveContextEntry(turns, entry).turn.provenance.type === 'derived' &&
            blockIds.length !== blocks.length
        ) {
            throw new Error('Derived replacement is an indivisible lineage unit without per-block source slices');
        }
        const chosen = new Set(blockIds);
        if (
            chosen.size !== blockIds.length ||
            blocks.filter((block) => chosen.has(block.id)).length !== blockIds.length ||
            canonicalJsonContentString(blocks.filter((block) => chosen.has(block.id)).map((block) => block.id)) !==
                canonicalJsonContentString(blockIds)
        ) {
            throw new Error('Partial context selection blocks are unavailable, duplicated or out of order');
        }
        removed.push(blockIds.length === blocks.length ? entry : { ...entry, block_ids: blockIds });
        const remainder = blocks.filter((block) => !chosen.has(block.id)).map((block) => block.id);
        if (remainder.length) retained.push({ ...entry, block_ids: remainder });
        let run: { entry: ContextEntry; selected: boolean; block_ids: string[] } | undefined;
        for (const block of blocks) {
            const isSelected = chosen.has(block.id);
            if (!run || run.selected !== isSelected) {
                run = { entry, selected: isSelected, block_ids: [] };
                segments.push(run);
            }
            run.block_ids.push(block.id);
            if (isSelected && !inRange) ranges += 1;
            inRange = isSelected;
        }
    }
    return { removed, retained, segments, ranges };
}
