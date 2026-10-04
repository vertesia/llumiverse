import { canonicalJsonContentString } from './content-integrity.js';
import { fingerprintJson } from './identity.js';
import type { RequestSourceProjectedTurn } from './request-source-view.js';
import { ContextChangePlanSchema } from './schemas/context-change.js';
import type {
    Asset,
    ContentBlock,
    ContextChangePlan,
    ContextChangePlanInput,
    ContextEntry,
    ConversationContext,
    ConversationDocument,
    ConversationRef,
} from './types.js';

/** Ephemeral authenticated active records. Completeness is established by the owning adapter;
 * this is not a conversation document, a history representation or publication authority.
 */
export interface ContextChangeActiveTurn {
    header: RequestSourceProjectedTurn['header'];
    active_blocks: ContentBlock[];
}
export interface ContextChangeWorkingSet {
    source: ConversationRef;
    context: ConversationContext;
    turns: ReadonlyMap<string, ContextChangeActiveTurn>;
    assets: Record<string, Asset>;
}

/** The materialized adapter preserves all original blocks and replacement identities. */
export function materializedContextChangeWorkingSet(document: ConversationDocument): ContextChangeWorkingSet {
    const turns = [
        ...document.turns,
        ...Object.values(document.compactions).flatMap((record) => record.replacement_turns),
    ];
    return {
        source: { conversation_id: document.id, revision: document.revision },
        context: document.context,
        assets: document.assets,
        turns: new Map(
            turns.map((turn): [string, ContextChangeActiveTurn] => {
                const { blocks, ...header } = turn;
                return [turn.id, { header, active_blocks: blocks }];
            }),
        ),
    };
}

function resolveWorkingSetEntry(turns: ContextChangeWorkingSet['turns'], entry: ContextEntry) {
    const turn = turns.get(entry.turn_id);
    if (!turn) throw new Error(`Context entry ${entry.id} references unavailable active turn ${entry.turn_id}`);
    const ids = entry.block_ids === undefined ? undefined : new Set(entry.block_ids);
    return { turn, blocks: turn.active_blocks.filter((block) => ids === undefined || ids.has(block.id)) };
}

export function selectedBlocks(turn: ContextChangeActiveTurn, entry: ContextEntry): ContentBlock[] {
    const ids = entry.block_ids === undefined ? undefined : new Set(entry.block_ids);
    return turn.active_blocks.filter((block) => ids === undefined || ids.has(block.id));
}

/** A partial selection partitions only the already-active top-level blocks, in original order. */
export function partitionSelection(
    document: ContextChangeWorkingSet,
    input: ContextChangePlanInput,
    entries = document.context.entries,
) {
    const turns = document.turns;
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
        const blocks = resolveWorkingSetEntry(turns, entry).blocks;
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
    // Ordinary replay is a retained wire representation, not immutable reasoning authority. A partial
    // semantic edit may retire an explicitly discardable replay in THAT original entry's remainder.
    // Keep its full payload in the original turn; only the active block selection changes. Replay in
    // another entry and nested replay still go through the unchanged dependency-closure guard.
    const changed = selectedIdentities(document, removed);
    const discardedReplayIds = new Set<string>();
    for (const segment of segments) {
        if (segment.selected || segment.block_ids === undefined || segment.block_ids.length === 0) continue;
        const blocks = resolveWorkingSetEntry(turns, segment.entry).blocks;
        for (const block of blocks) {
            if (
                !segment.block_ids.includes(block.id) ||
                block.type !== 'native_replay' ||
                block.dependency_policy !== 'discard_on_dependency_change' ||
                !(
                    block.dependencies.turn_ids.some((id) => changed.turnIds.has(id)) ||
                    block.dependencies.block_ids.some((id) => changed.blockIds.has(id)) ||
                    block.dependencies.call_ids.some((id) => changed.callIds.has(id))
                )
            )
                continue;
            if (document.context.protected_entry_ids.includes(segment.entry.id))
                throw new Error('Context change selects a protected entry');
            discardedReplayIds.add(block.id);
        }
        segment.block_ids = segment.block_ids.filter((id) => !discardedReplayIds.has(id));
    }
    const activeRetained = retained.flatMap((entry) => {
        if (entry.block_ids === undefined) return [entry];
        const remaining = entry.block_ids.filter((id) => !discardedReplayIds.has(id));
        return remaining.length ? [{ ...entry, block_ids: remaining }] : [];
    });
    return { removed, retained: activeRetained, segments, ranges, discardedReplayIds };
}

/** Each replacement maps to one ordered selected range; intervening content remains in place. */
export function selectedRanges(document: ContextChangeWorkingSet, partition: ReturnType<typeof partitionSelection>) {
    const turns = document.turns;
    const ranges: { entry_ids: string[]; turn_ids: string[]; block_ids: string[]; blocks: ContentBlock[] }[] = [];
    let open = false;
    for (const segment of partition.segments) {
        if (!segment.selected) {
            open = false;
            continue;
        }
        if (!open) {
            ranges.push({ entry_ids: [], turn_ids: [], block_ids: [], blocks: [] });
            open = true;
        }
        const range = ranges[ranges.length - 1];
        if (!range.entry_ids.includes(segment.entry.id)) range.entry_ids.push(segment.entry.id);
        if (!range.turn_ids.includes(segment.entry.turn_id)) range.turn_ids.push(segment.entry.turn_id);
        const blocks = resolveWorkingSetEntry(turns, segment.entry).blocks;
        const ids = segment.block_ids === undefined ? undefined : new Set(segment.block_ids);
        for (const block of blocks) {
            if (ids !== undefined && !ids.has(block.id)) continue;
            range.block_ids.push(block.id);
            range.blocks.push(block);
        }
    }
    return ranges;
}

export function collectAssetIds(blocks: readonly ContentBlock[]): Set<string> {
    const ids = new Set<string>();
    const visit = (block: ContentBlock): void => {
        if ('asset_id' in block) ids.add(block.asset_id);
        if (block.type === 'tool_call' && block.arguments.type === 'externalized_json') {
            for (const item of block.arguments.hydration) ids.add(item.asset_id);
        }
        if (block.type === 'tool_result') for (const nested of block.content) visit(nested);
    };
    for (const block of blocks) visit(block);
    return ids;
}

function selectedIdentities(document: ContextChangeWorkingSet, entries: readonly ContextEntry[]) {
    const turns = document.turns;
    const turnIds = new Set<string>();
    const blockIds = new Set<string>();
    const callIds = new Set<string>();
    const calls = new Set<string>();
    const results = new Set<string>();
    const replay: Extract<ContentBlock, { type: 'native_replay' }>[] = [];
    const visit = (block: ContentBlock): void => {
        blockIds.add(block.id);
        if (block.type === 'tool_call') {
            callIds.add(block.call_id);
            if (block.executor === 'application') calls.add(block.call_id);
        }
        if (block.type === 'tool_result') {
            callIds.add(block.call_id);
            results.add(block.call_id);
            for (const nested of block.content) visit(nested);
        }
        if (block.type === 'native_replay') replay.push(block);
    };
    for (const entry of entries) {
        const turn = resolveWorkingSetEntry(turns, entry).turn;
        turnIds.add(turn.header.id);
        for (const block of selectedBlocks(turn, entry)) visit(block);
    }
    return { turnIds, blockIds, callIds, calls, results, replay };
}

export function assertContextMutationDependencyClosure(
    document: ContextChangeWorkingSet,
    removed: readonly ContextEntry[],
    retained: readonly ContextEntry[],
): void {
    const removedIds = new Set(removed.map((entry) => entry.id));
    if (document.context.protected_entry_ids.some((id) => removedIds.has(id))) {
        throw new Error('Context change selects a protected entry');
    }
    const turns = document.turns;
    for (const entry of removed) {
        const turn = resolveWorkingSetEntry(turns, entry).turn;
        if (turn.header.authority === 'system' || turn.header.authority === 'developer') {
            throw new Error(`Context change selects protected ${turn.header.authority} turn ${turn.header.id}`);
        }
    }
    const selected = selectedIdentities(document, removed);
    const before = selectedIdentities(document, document.context.entries);
    const after = selectedIdentities(document, retained);
    // Partial remainder construction can retire an explicitly discardable wire unit as well as the
    // nominated semantic blocks. Every disappearing active block participates in dependency closure;
    // a protected or separate discardable dependent must not become an implicit cascading edit.
    for (const id of before.blockIds) {
        if (!after.blockIds.has(id)) selected.blockIds.add(id);
    }
    if (selected.replay.length > 0) throw new Error('Context change selects a protected native replay unit');
    for (const callId of selected.calls) {
        if (!before.results.has(callId)) throw new Error(`Context change selects pending tool call ${callId}`);
    }
    for (const callId of new Set([...before.calls, ...before.results])) {
        if (!before.calls.has(callId) || !before.results.has(callId)) continue;
        if (after.calls.has(callId) !== after.results.has(callId)) {
            throw new Error(`Context change would split application tool exchange ${callId}`);
        }
    }
    for (const block of after.replay) {
        const dependencies = block.dependencies;
        if (
            dependencies.turn_ids.some((id) => selected.turnIds.has(id)) ||
            dependencies.block_ids.some((id) => selected.blockIds.has(id)) ||
            dependencies.call_ids.some((id) => selected.callIds.has(id))
        ) {
            throw new Error(`Context change would orphan native replay dependency ${block.id}`);
        }
    }
}

/** Shared exact planning math after the adapter proves active dependency completeness. */
export async function planContextChangeWorkingSet(
    document: ContextChangeWorkingSet,
    ownedInput: ContextChangePlanInput,
): Promise<ContextChangePlan> {
    const entryIds = ownedInput.entry_ids;
    if (
        ownedInput.expected_revision !== document.source.revision ||
        ownedInput.expected_context_revision !== document.context.revision
    ) {
        throw new Error('Context change revision conflict');
    }
    if (entryIds.length === 0 || new Set(entryIds).size !== entryIds.length) {
        throw new Error('Context change requires unique selected entry IDs');
    }
    const partition = partitionSelection(document, ownedInput);
    const { removed, retained, ranges } = partition;
    assertContextMutationDependencyClosure(document, removed, retained);
    const turns = document.turns;
    const source = removed.map((entry) => {
        const turn = resolveWorkingSetEntry(turns, entry).turn;
        const turnIdentity = turn.header;
        return { entry, turn: turnIdentity, blocks: selectedBlocks(turn, entry) };
    });
    const sourceTurnIds = [...new Set(removed.map((entry) => entry.turn_id))];
    const sourceBlockIds = removed.some((entry) => entry.block_ids !== undefined)
        ? [...new Set(source.flatMap(({ blocks }) => blocks.map((block) => block.id)))]
        : [];
    const assetIds = collectAssetIds(source.flatMap(({ blocks }) => blocks));
    const queue = [...assetIds];
    for (let index = 0; index < queue.length; index += 1) {
        const asset = document.assets[queue[index]];
        if (asset?.provenance.type === 'derived' && !assetIds.has(asset.provenance.source_asset_id)) {
            assetIds.add(asset.provenance.source_asset_id);
            queue.push(asset.provenance.source_asset_id);
        }
    }
    const assets = [...assetIds].sort().map((id) => ({ id, asset: document.assets[id] }));
    return ContextChangePlanSchema.parse({
        source_fingerprint: await fingerprintJson({
            conversation_id: document.source.conversation_id,
            revision: document.source.revision,
            context_revision: document.context.revision,
            selected: source,
            assets,
            ...(partition.discardedReplayIds.size
                ? {
                      discarded_replay: [...document.turns.values()].flatMap((turn) =>
                          turn.active_blocks.filter((block) => partition.discardedReplayIds.has(block.id)),
                      ),
                  }
                : {}),
        }),
        ...(partition.discardedReplayIds.size ? { discarded_replay_block_ids: [...partition.discardedReplayIds] } : {}),
        entry_ids: removed.map((entry) => entry.id),
        ...(Object.hasOwn(ownedInput, 'selected_block_ids')
            ? { selected_block_ids: ownedInput.selected_block_ids }
            : {}),
        ...(Object.hasOwn(ownedInput, 'selected_entries') ? { selected_entries: ownedInput.selected_entries } : {}),
        source_turn_ids: sourceTurnIds,
        source_block_ids: sourceBlockIds,
        selected_asset_ids: [...assetIds].sort(),
        disjoint_ranges: ranges,
    });
}
