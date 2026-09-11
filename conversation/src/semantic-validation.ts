import { boundConversationDiagnostic, diagnosticPointer, diagnosticValue } from './diagnostics.js';
import { UsageMetricSchema } from './schemas/execution.js';
import type {
    Asset,
    CompactionRecord,
    ContentBlock,
    ContextEntry,
    ConversationDiagnostic,
    ConversationDocument,
    ConversationTurn,
    Generation,
    GenerationUsage,
    NativeReplayBlock,
    NestedToolResultContentBlock,
    RequestReceipt,
    SemanticConversationDiagnostic,
    SemanticConversationDiagnosticCode,
    ToolCallBlock,
    ToolResultBlock,
} from './types.js';

const MAX_SEMANTIC_DIAGNOSTICS = 256;
const USAGE_METRICS = UsageMetricSchema.options;

type AnyBlock = ContentBlock | NestedToolResultContentBlock;

interface LocatedTurn {
    turn: ConversationTurn;
    path: string;
    source: boolean;
    compaction_id?: string;
}

interface LocatedBlock {
    block: AnyBlock;
    turn_id: string;
    path: string;
}

interface LocatedCall {
    block: ToolCallBlock;
    turn_id: string;
    path: string;
    source: boolean;
}

interface ActiveSelection {
    turn_id: string;
    all_blocks: boolean;
    block_ids: ReadonlySet<string>;
    origin: 'direct' | 'replacement';
    path: string;
    compaction_id?: string;
}

interface ActiveSelectionIndex {
    first: ActiveSelection;
    all?: ActiveSelection;
    by_block_id: Map<string, ActiveSelection>;
}

interface GraphFrame {
    id: string;
    edges: readonly string[];
    next_edge: number;
}

function recordPath(collection: string, id: string): string {
    return diagnosticPointer([collection, id]);
}

function hasOwn(record: object, key: PropertyKey): boolean {
    return Object.hasOwn(record, key);
}

function selectionsOverlap(first: ActiveSelection, second: ActiveSelection): boolean {
    if (first.turn_id !== second.turn_id) {
        return false;
    }
    if (first.all_blocks || second.all_blocks) {
        return true;
    }
    for (const blockId of first.block_ids) {
        if (second.block_ids.has(blockId)) {
            return true;
        }
    }
    return false;
}

function validateDirectedCycles(
    nodeIds: Iterable<string>,
    edgesFor: (id: string) => readonly string[],
    onCycle: (from: string, to: string) => void,
): void {
    const colors = new Map<string, 'visiting' | 'visited'>();
    for (const start of nodeIds) {
        if (colors.has(start)) {
            continue;
        }
        colors.set(start, 'visiting');
        const stack: GraphFrame[] = [{ id: start, edges: edgesFor(start), next_edge: 0 }];
        while (stack.length > 0) {
            const frame = stack[stack.length - 1];
            if (frame.next_edge >= frame.edges.length) {
                colors.set(frame.id, 'visited');
                stack.pop();
                continue;
            }
            const target = frame.edges[frame.next_edge];
            frame.next_edge += 1;
            const targetColor = colors.get(target);
            if (targetColor === 'visiting') {
                onCycle(frame.id, target);
            } else if (targetColor === undefined) {
                colors.set(target, 'visiting');
                stack.push({ id: target, edges: edgesFor(target), next_edge: 0 });
            }
        }
    }
}

function compareTimestamps(first: string, second: string): number {
    return Date.parse(first) - Date.parse(second);
}

function getTurnGenerationId(turn: ConversationTurn): string | undefined {
    return turn.kind === 'agent' && 'generation_id' in turn ? turn.generation_id : undefined;
}

function allTurnBlocks(turn: ConversationTurn, turnPath: string): LocatedBlock[] {
    const blocks: LocatedBlock[] = [];
    for (let blockIndex = 0; blockIndex < turn.blocks.length; blockIndex += 1) {
        const block = turn.blocks[blockIndex];
        const path = `${turnPath}/blocks/${blockIndex}`;
        blocks.push({ block, turn_id: turn.id, path });
        if (block.type === 'tool_result') {
            for (let nestedIndex = 0; nestedIndex < block.content.length; nestedIndex += 1) {
                blocks.push({
                    block: block.content[nestedIndex],
                    turn_id: turn.id,
                    path: `${path}/content/${nestedIndex}`,
                });
            }
        }
    }
    return blocks;
}

function authorityRank(authority: ConversationTurn['authority']): number {
    if (authority === 'system') {
        return 2;
    }
    if (authority === 'developer') {
        return 1;
    }
    return 0;
}

function validateUsage(
    usage: GenerationUsage,
    path: string,
    add: (code: SemanticConversationDiagnosticCode, path: string, message: string, recordId?: string) => void,
    generationId: string,
): void {
    const normalizedMetrics = USAGE_METRICS.filter((metric) => usage[metric] !== undefined);
    const provenance = usage.accounting_provenance;
    const provenanceMetrics = provenance === undefined ? [] : USAGE_METRICS.filter((metric) => provenance[metric]);

    if (
        normalizedMetrics.length === 0 &&
        provenanceMetrics.length === 0 &&
        (usage.reported_usage?.length ?? 0) === 0 &&
        usage.cost === undefined
    ) {
        add(
            'GENERATION_USAGE_EMPTY',
            path,
            'Unknown usage must be absent instead of represented by an empty object',
            generationId,
        );
    }

    for (const metric of USAGE_METRICS) {
        const hasValue = usage[metric] !== undefined;
        const hasProvenance = provenance?.[metric] !== undefined;
        if (hasValue !== hasProvenance) {
            add(
                'ACCOUNTING_PROVENANCE_MISMATCH',
                `${path}/${metric}`,
                `Normalized ${metric} and its accounting provenance must either both be present or both be absent`,
                generationId,
            );
        }
    }

    const input = usage.input_tokens;
    const output = usage.output_tokens;
    const total = usage.total_tokens;
    const inputBasis = provenance?.input_tokens?.accounting_basis;
    const outputBasis = provenance?.output_tokens?.accounting_basis;
    const compatibleInputAndOutput =
        input !== undefined && output !== undefined && inputBasis !== undefined && inputBasis === outputBasis;
    if (total !== undefined && !compatibleInputAndOutput) {
        add(
            'USAGE_TOTAL_INVALID',
            `${path}/total_tokens`,
            'Canonical total_tokens requires compatible input_tokens and output_tokens',
            generationId,
        );
    }
    if (compatibleInputAndOutput) {
        const sum = input + output;
        if (!Number.isSafeInteger(sum)) {
            add(
                'USAGE_OVERFLOW',
                `${path}/total_tokens`,
                'input_tokens plus output_tokens exceeds Number.MAX_SAFE_INTEGER',
                generationId,
            );
        } else if (total === undefined || total !== sum || provenance?.total_tokens?.accounting_basis !== inputBasis) {
            add(
                'USAGE_TOTAL_INVALID',
                `${path}/total_tokens`,
                'Canonical total_tokens must equal input_tokens plus output_tokens when both are known',
                generationId,
            );
        }
    }

    if (input !== undefined) {
        if (usage.cache_read_tokens !== undefined && usage.cache_read_tokens > input) {
            add(
                'USAGE_BREAKDOWN_INVALID',
                `${path}/cache_read_tokens`,
                'cache_read_tokens cannot exceed input_tokens',
                generationId,
            );
        }
        if (usage.cache_write_tokens !== undefined && usage.cache_write_tokens > input) {
            add(
                'USAGE_BREAKDOWN_INVALID',
                `${path}/cache_write_tokens`,
                'cache_write_tokens cannot exceed input_tokens',
                generationId,
            );
        }
    }
    if (output !== undefined && usage.reasoning_tokens !== undefined && usage.reasoning_tokens > output) {
        add(
            'USAGE_BREAKDOWN_INVALID',
            `${path}/reasoning_tokens`,
            'reasoning_tokens cannot exceed output_tokens',
            generationId,
        );
    }

    if (usage.input_partition !== undefined) {
        const newInput = usage.input_new_tokens;
        const cacheRead = usage.cache_read_tokens;
        const cacheWrite = usage.cache_write_tokens;
        const includesWrite = usage.input_partition.cache_write_bucket === 'included';
        const partitionBases = [
            provenance?.input_new_tokens?.accounting_basis,
            provenance?.cache_read_tokens?.accounting_basis,
            ...(includesWrite ? [provenance?.cache_write_tokens?.accounting_basis] : []),
        ];
        if (
            input === undefined ||
            newInput === undefined ||
            cacheRead === undefined ||
            (includesWrite && cacheWrite === undefined) ||
            (!includesWrite && cacheWrite !== undefined) ||
            inputBasis === undefined ||
            partitionBases.some((basis) => basis !== inputBasis)
        ) {
            add(
                'INPUT_PARTITION_INVALID',
                `${path}/input_partition`,
                'A complete disjoint input partition must declare exactly its applicable token buckets',
                generationId,
            );
        } else {
            const partitionSum = newInput + cacheRead + (cacheWrite ?? 0);
            if (!Number.isSafeInteger(partitionSum)) {
                add(
                    'USAGE_OVERFLOW',
                    `${path}/input_partition`,
                    'Input partition token buckets exceed Number.MAX_SAFE_INTEGER when added',
                    generationId,
                );
            } else if (partitionSum !== input) {
                add(
                    'INPUT_PARTITION_INVALID',
                    `${path}/input_partition`,
                    'Complete disjoint input token buckets must add up to input_tokens',
                    generationId,
                );
            }
        }
    }
    if (
        usage.input_new_tokens !== undefined &&
        provenance?.input_new_tokens?.method === 'derived' &&
        usage.input_partition === undefined
    ) {
        add(
            'INPUT_PARTITION_INVALID',
            `${path}/input_new_tokens`,
            'Derived input_new_tokens requires a declared complete disjoint input partition',
            generationId,
        );
    }
}

function validateRequestReceipt(
    receipt: RequestReceipt,
    generation: Generation,
    path: string,
    document: ConversationDocument,
    turnsById: Map<string, LocatedTurn>,
    blocksById: Map<string, LocatedBlock>,
    callsById: Map<string, LocatedCall>,
    add: (code: SemanticConversationDiagnosticCode, path: string, message: string, recordId?: string) => void,
): void {
    if (receipt.source.conversation_id !== document.id || receipt.source.revision > document.revision) {
        add(
            'GENERATION_SOURCE_INVALID',
            `${path}/source`,
            'Request receipt source is not a valid revision of this document',
        );
    }
    if (
        generation.record_source === 'executed' &&
        (receipt.request_id !== generation.request_id || receipt.attempt_id !== generation.attempt_id)
    ) {
        add(
            'GENERATION_REQUEST_MISMATCH',
            path,
            'Request receipt identity must match its executed generation',
            generation.id,
        );
    }
    if (receipt.source_tail_turn_id !== undefined) {
        const retainedTail = turnsById.get(receipt.source_tail_turn_id);
        if (retainedTail !== undefined && !retainedTail.source) {
            add('REQUEST_MAPPING_INVALID', `${path}/source_tail_turn_id`, 'Retained request tail is not a source turn');
        }
    }
    // Request receipts are historical facts and intentionally outlive transcript, definition, and
    // asset deletion. Bindings are checked against retained records, but an absent historical record
    // is not interpreted as a dangling live-document reference.
    for (let index = 0; index < receipt.asset_versions.length; index += 1) {
        const binding = receipt.asset_versions[index];
        const asset = hasOwn(document.assets, binding.asset_id) ? document.assets[binding.asset_id] : undefined;
        if (asset?.content_hash !== undefined && asset.content_hash !== binding.content_hash) {
            add(
                'REQUEST_MAPPING_INVALID',
                `${path}/asset_versions/${index}/content_hash`,
                `Asset ${diagnosticValue(binding.asset_id)} does not match the receipt content hash`,
            );
        }
    }
    for (let index = 0; index < receipt.item_mappings.length; index += 1) {
        const mapping = receipt.item_mappings[index];
        const exists =
            (mapping.kind === 'turn' && turnsById.has(mapping.canonical_id)) ||
            (mapping.kind === 'block' && blocksById.has(mapping.canonical_id)) ||
            (mapping.kind === 'call' && callsById.has(mapping.canonical_id));
        const retainedWithAnotherKind =
            turnsById.has(mapping.canonical_id) ||
            blocksById.has(mapping.canonical_id) ||
            callsById.has(mapping.canonical_id);
        if (!exists && retainedWithAnotherKind) {
            add(
                'REQUEST_MAPPING_INVALID',
                `${path}/item_mappings/${index}/canonical_id`,
                `Retained canonical item ${diagnosticValue(mapping.canonical_id)} does not have kind ${mapping.kind}`,
            );
        }
    }
}

function validateMediaSelection(
    block: AnyBlock,
    blockPath: string,
    asset: Asset,
    add: (code: SemanticConversationDiagnosticCode, path: string, message: string, recordId?: string) => void,
): void {
    if (block.type === 'image' && block.selection?.coordinate_space === 'normalized') {
        if (block.selection.x + block.selection.width > 1 || block.selection.y + block.selection.height > 1) {
            add(
                'SELECTION_RANGE_INVALID',
                `${blockPath}/selection`,
                'Normalized image selection must fit within the unit square',
            );
        }
    }
    if (block.type === 'image' && block.selection?.coordinate_space === 'pixels' && asset.media !== undefined) {
        if (
            (asset.media.width !== undefined && block.selection.x + block.selection.width > asset.media.width) ||
            (asset.media.height !== undefined && block.selection.y + block.selection.height > asset.media.height)
        ) {
            add(
                'SELECTION_RANGE_INVALID',
                `${blockPath}/selection`,
                'Pixel image selection exceeds known asset dimensions',
            );
        }
    }
    if (block.type === 'document' && block.selection !== undefined) {
        if (block.selection.from_page > block.selection.through_page) {
            add('SELECTION_RANGE_INVALID', `${blockPath}/selection`, 'Document page range is reversed');
        }
        if (asset.media?.page_count !== undefined && block.selection.through_page > asset.media.page_count) {
            add(
                'SELECTION_RANGE_INVALID',
                `${blockPath}/selection`,
                'Document page range exceeds the known page count',
            );
        }
    }
    if ((block.type === 'audio' || block.type === 'video') && block.selection !== undefined) {
        if (block.selection.start_seconds >= block.selection.end_seconds) {
            add('SELECTION_RANGE_INVALID', `${blockPath}/selection`, 'Media time range must be nonempty and ordered');
        }
        if (asset.media?.duration_seconds !== undefined && block.selection.end_seconds > asset.media.duration_seconds) {
            add('SELECTION_RANGE_INVALID', `${blockPath}/selection`, 'Media time range exceeds the known duration');
        }
    }
}

function validateReplayDependencies(
    replay: NativeReplayBlock,
    path: string,
    turnsById: Map<string, LocatedTurn>,
    blocksById: Map<string, LocatedBlock>,
    callsById: Map<string, LocatedCall>,
    requestIds: Set<string>,
    add: (code: SemanticConversationDiagnosticCode, path: string, message: string, recordId?: string) => void,
): void {
    const groups: Array<[readonly string[], { has(id: string): boolean }, string]> = [
        [replay.dependencies.turn_ids, turnsById, 'turn_ids'],
        [replay.dependencies.block_ids, blocksById, 'block_ids'],
        [replay.dependencies.call_ids, callsById, 'call_ids'],
        [replay.dependencies.request_ids, requestIds, 'request_ids'],
    ];
    for (const [ids, known, field] of groups) {
        for (let index = 0; index < ids.length; index += 1) {
            if (!known.has(ids[index])) {
                add(
                    'REFERENCE_NOT_FOUND',
                    `${path}/dependencies/${field}/${index}`,
                    `Replay dependency ${diagnosticValue(ids[index])} is missing`,
                );
            }
        }
    }
}

function validateCompaction(
    compaction: CompactionRecord,
    path: string,
    sourceTurnsById: Map<string, LocatedTurn>,
    blocksById: Map<string, LocatedBlock>,
    document: ConversationDocument,
    add: (code: SemanticConversationDiagnosticCode, path: string, message: string, recordId?: string) => void,
): void {
    const sourceAuthorities = new Set<ConversationTurn['authority']>();
    const declaredSourceTurns = new Set(compaction.source.turn_ids);
    for (let index = 0; index < compaction.source.turn_ids.length; index += 1) {
        const turnId = compaction.source.turn_ids[index];
        const located = sourceTurnsById.get(turnId);
        if (located === undefined || located.compaction_id === compaction.id) {
            add(
                'REFERENCE_NOT_FOUND',
                `${path}/source/turn_ids/${index}`,
                `Compaction source turn ${diagnosticValue(turnId)} is missing or belongs to the same compaction`,
            );
        } else {
            sourceAuthorities.add(located.turn.authority);
        }
    }
    for (let index = 0; index < (compaction.source.block_ids?.length ?? 0); index += 1) {
        const blockId = compaction.source.block_ids?.[index];
        const located = blockId === undefined ? undefined : blocksById.get(blockId);
        if (located === undefined || !declaredSourceTurns.has(located.turn_id)) {
            add(
                'REFERENCE_NOT_FOUND',
                `${path}/source/block_ids/${index}`,
                `Compaction block ${diagnosticValue(blockId ?? '')} is outside the declared source turns`,
            );
        }
    }
    if (sourceAuthorities.size > 1) {
        add(
            'DERIVED_AUTHORITY_MIXED',
            `${path}/source/turn_ids`,
            'One compaction replacement cannot combine mixed source authorities',
            compaction.id,
        );
    }
    const sourceAuthority = sourceAuthorities.values().next().value;
    const declaredSourceBlocks =
        compaction.source.block_ids === undefined ? undefined : new Set(compaction.source.block_ids);
    for (let index = 0; index < compaction.replacement_turns.length; index += 1) {
        const replacement = compaction.replacement_turns[index];
        const replacementPath = `${path}/replacement_turns/${index}`;
        if (replacement.provenance.type !== 'derived' || replacement.provenance.derivation_id !== compaction.id) {
            add(
                'DERIVED_PROVENANCE_MISMATCH',
                `${replacementPath}/provenance`,
                'Compaction replacement must be derived by the containing compaction',
                replacement.id,
            );
            continue;
        }
        for (let sourceIndex = 0; sourceIndex < replacement.provenance.source_turn_ids.length; sourceIndex += 1) {
            const sourceTurnId = replacement.provenance.source_turn_ids[sourceIndex];
            if (!declaredSourceTurns.has(sourceTurnId)) {
                add(
                    'DERIVED_PROVENANCE_MISMATCH',
                    `${replacementPath}/provenance/source_turn_ids/${sourceIndex}`,
                    `Derived source turn ${diagnosticValue(sourceTurnId)} is outside the compaction selection`,
                    replacement.id,
                );
            }
        }
        if (declaredSourceBlocks !== undefined && replacement.provenance.source_block_ids === undefined) {
            add(
                'DERIVED_PROVENANCE_MISMATCH',
                `${replacementPath}/provenance/source_block_ids`,
                'A replacement of a partial block selection must identify its source blocks',
                replacement.id,
            );
        }
        const replacementSourceTurns = new Set(replacement.provenance.source_turn_ids);
        for (
            let sourceIndex = 0;
            sourceIndex < (replacement.provenance.source_block_ids?.length ?? 0);
            sourceIndex += 1
        ) {
            const sourceBlockId = replacement.provenance.source_block_ids?.[sourceIndex];
            const located = sourceBlockId === undefined ? undefined : blocksById.get(sourceBlockId);
            if (
                located === undefined ||
                !replacementSourceTurns.has(located.turn_id) ||
                !declaredSourceTurns.has(located.turn_id) ||
                (declaredSourceBlocks !== undefined && !declaredSourceBlocks.has(sourceBlockId ?? ''))
            ) {
                add(
                    'DERIVED_PROVENANCE_MISMATCH',
                    `${replacementPath}/provenance/source_block_ids/${sourceIndex}`,
                    `Derived block ${diagnosticValue(sourceBlockId ?? '')} is outside the declared source selection`,
                    replacement.id,
                );
            }
        }
        if (sourceAuthority !== undefined && authorityRank(replacement.authority) > authorityRank(sourceAuthority)) {
            add(
                'DERIVED_AUTHORITY_PROMOTED',
                `${replacementPath}/authority`,
                'Derived replacement authority cannot exceed its source authority',
                replacement.id,
            );
        }
    }
    for (let index = 0; index < compaction.retained_asset_ids.length; index += 1) {
        const assetId = compaction.retained_asset_ids[index];
        if (!hasOwn(document.assets, assetId)) {
            add(
                'REFERENCE_NOT_FOUND',
                `${path}/retained_asset_ids/${index}`,
                `Retained asset ${diagnosticValue(assetId)} does not exist`,
            );
        }
    }
    for (let index = 0; index < compaction.generation_ids.length; index += 1) {
        const generationId = compaction.generation_ids[index];
        if (!hasOwn(document.generations, generationId)) {
            add(
                'REFERENCE_NOT_FOUND',
                `${path}/generation_ids/${index}`,
                `Generation ${diagnosticValue(generationId)} does not exist`,
            );
        }
    }
}

export function validateConversationSemantics(document: ConversationDocument): ConversationDiagnostic[] {
    const diagnostics: ConversationDiagnostic[] = [];
    const add = (code: SemanticConversationDiagnosticCode, path: string, message: string, recordId?: string): void => {
        if (diagnostics.length < MAX_SEMANTIC_DIAGNOSTICS - 1) {
            const diagnostic: SemanticConversationDiagnostic = {
                code,
                stage: 'semantic',
                path,
                message,
                record_id: recordId,
            };
            diagnostics.push(boundConversationDiagnostic(diagnostic));
        } else if (diagnostics.length === MAX_SEMANTIC_DIAGNOSTICS - 1) {
            diagnostics.push(
                boundConversationDiagnostic({
                    code: 'SEMANTIC_DIAGNOSTIC_LIMIT',
                    stage: 'semantic',
                    path,
                    message: `Semantic validation stopped reporting after ${MAX_SEMANTIC_DIAGNOSTICS - 1} diagnostics`,
                    limit: MAX_SEMANTIC_DIAGNOSTICS - 1,
                }),
            );
        }
    };

    const globalIds = new Map<string, { kind: string; path: string }>();
    const registerId = (id: string, kind: string, path: string): void => {
        const prior = globalIds.get(id);
        if (prior !== undefined) {
            add(
                'DUPLICATE_ID',
                path,
                `${kind} ID ${diagnosticValue(id)} duplicates ${prior.kind} at ${prior.path}`,
                id,
            );
        } else {
            globalIds.set(id, { kind, path });
        }
    };
    registerId(document.id, 'conversation', '/id');

    if (compareTimestamps(document.created_at, document.updated_at) > 0) {
        add('TIMESTAMP_ORDER_INVALID', '/updated_at', 'updated_at cannot be earlier than created_at', document.id);
    }
    if (document.context.revision > document.revision) {
        add(
            'CONTEXT_REVISION_INVALID',
            '/context/revision',
            'Context revision cannot exceed document revision',
            document.id,
        );
    }

    const turnsById = new Map<string, LocatedTurn>();
    const blocksById = new Map<string, LocatedBlock>();
    const callsById = new Map<string, LocatedCall>();
    const sourceResultCalls = new Map<string, string>();
    const requestIds = new Set<string>();

    const registerTurn = (turn: ConversationTurn, path: string, source: boolean, compactionId?: string): void => {
        registerId(turn.id, source ? 'source turn' : 'replacement turn', `${path}/id`);
        if (!turnsById.has(turn.id)) {
            turnsById.set(turn.id, { turn, path, source, compaction_id: compactionId });
        }
        const blocks = allTurnBlocks(turn, path);
        for (const located of blocks) {
            registerId(located.block.id, 'block', `${located.path}/id`);
            if (!blocksById.has(located.block.id)) {
                blocksById.set(located.block.id, located);
            }
            if (located.block.type === 'tool_call') {
                registerId(located.block.call_id, 'tool call', `${located.path}/call_id`);
                if (!callsById.has(located.block.call_id)) {
                    callsById.set(located.block.call_id, { ...located, block: located.block, source });
                }
            }
            if (source && located.block.type === 'tool_result') {
                const priorResult = sourceResultCalls.get(located.block.call_id);
                if (priorResult !== undefined) {
                    add(
                        'DUPLICATE_TERMINAL_RESULT',
                        `${located.path}/call_id`,
                        `Call ${diagnosticValue(located.block.call_id)} already has terminal result at ${priorResult}`,
                        located.block.call_id,
                    );
                } else {
                    sourceResultCalls.set(located.block.call_id, located.path);
                }
            }
        }
    };

    for (let index = 0; index < document.turns.length; index += 1) {
        registerTurn(document.turns[index], `/turns/${index}`, true);
    }
    for (const [compactionId, compaction] of Object.entries(document.compactions)) {
        const path = recordPath('compactions', compactionId);
        if (compactionId !== compaction.id) {
            add(
                'MAP_KEY_ID_MISMATCH',
                path,
                `Compaction key ${diagnosticValue(compactionId)} differs from ID ${diagnosticValue(compaction.id)}`,
            );
        }
        registerId(compaction.id, 'compaction', `${path}/id`);
        for (let index = 0; index < compaction.replacement_turns.length; index += 1) {
            registerTurn(
                compaction.replacement_turns[index],
                `${path}/replacement_turns/${index}`,
                false,
                compaction.id,
            );
        }
    }

    const checkEntityMap = <T extends { id: string }>(
        collection: string,
        record: Record<string, T>,
        kind: string,
    ): void => {
        for (const [key, value] of Object.entries(record)) {
            const path = recordPath(collection, key);
            if (key !== value.id) {
                add(
                    'MAP_KEY_ID_MISMATCH',
                    path,
                    `${kind} map key ${diagnosticValue(key)} does not match id ${diagnosticValue(value.id)}`,
                    value.id,
                );
            }
            registerId(value.id, kind, `${path}/id`);
        }
    };

    checkEntityMap('generations', document.generations, 'generation');
    checkEntityMap('operation_receipts', document.operation_receipts, 'operation receipt');
    checkEntityMap('execution_receipts', document.execution_receipts, 'execution receipt');
    checkEntityMap('assets', document.assets, 'asset');
    checkEntityMap('tool_definitions', document.tool_definitions, 'tool definition');

    for (const [generationId, generation] of Object.entries(document.generations)) {
        const path = recordPath('generations', generationId);
        if (generation.source.conversation_id !== document.id || generation.source.revision > document.revision) {
            add(
                'GENERATION_SOURCE_INVALID',
                `${path}/source`,
                'Generation source is not a valid revision of this document',
            );
        }
        if (generation.request_id !== undefined) {
            requestIds.add(generation.request_id);
        }
        if (generation.request_receipt !== undefined) {
            registerId(generation.request_receipt.id, 'request receipt', `${path}/request_receipt/id`);
            requestIds.add(generation.request_receipt.request_id);
            validateRequestReceipt(
                generation.request_receipt,
                generation,
                `${path}/request_receipt`,
                document,
                turnsById,
                blocksById,
                callsById,
                add,
            );
        }
        if (generation.usage !== undefined) {
            validateUsage(generation.usage, `${path}/usage`, add, generation.id);
        }
        if (
            generation.timestamps.started_at !== undefined &&
            generation.timestamps.completed_at !== undefined &&
            compareTimestamps(generation.timestamps.started_at, generation.timestamps.completed_at) > 0
        ) {
            add(
                'TIMESTAMP_ORDER_INVALID',
                `${path}/timestamps/completed_at`,
                'Generation completed_at cannot be earlier than started_at',
                generation.id,
            );
        }
    }

    for (const located of turnsById.values()) {
        const { turn, path } = located;
        const generationId = getTurnGenerationId(turn);
        if (generationId !== undefined && !hasOwn(document.generations, generationId)) {
            add(
                'REFERENCE_NOT_FOUND',
                `${path}/generation_id`,
                `Generation ${diagnosticValue(generationId)} does not exist`,
                turn.id,
            );
        }
        if (turn.parent_turn_id !== undefined && !turnsById.has(turn.parent_turn_id)) {
            add(
                'REFERENCE_NOT_FOUND',
                `${path}/parent_turn_id`,
                `Parent turn ${diagnosticValue(turn.parent_turn_id)} does not exist`,
                turn.id,
            );
        }
        if (turn.execution_id !== undefined && !hasOwn(document.execution_receipts, turn.execution_id)) {
            add(
                'REFERENCE_NOT_FOUND',
                `${path}/execution_id`,
                `Execution receipt ${diagnosticValue(turn.execution_id)} does not exist`,
                turn.id,
            );
        }
        if (
            turn.timestamps.started_at !== undefined &&
            turn.timestamps.completed_at !== undefined &&
            compareTimestamps(turn.timestamps.started_at, turn.timestamps.completed_at) > 0
        ) {
            add(
                'TIMESTAMP_ORDER_INVALID',
                `${path}/timestamps/completed_at`,
                'Turn completed_at cannot be earlier than started_at',
                turn.id,
            );
        }
    }

    validateDirectedCycles(
        turnsById.keys(),
        (turnId) => {
            const parentId = turnsById.get(turnId)?.turn.parent_turn_id;
            return parentId !== undefined && turnsById.has(parentId) ? [parentId] : [];
        },
        (from, to) => {
            const located = turnsById.get(from);
            add(
                'PARENT_TURN_CYCLE',
                `${located?.path ?? ''}/parent_turn_id`,
                `Parent-turn relationship contains a cycle through ${diagnosticValue(to)}`,
                from,
            );
        },
    );
    validateDirectedCycles(
        turnsById.keys(),
        (turnId) => {
            const turn = turnsById.get(turnId)?.turn;
            return turn?.provenance.type === 'derived'
                ? turn.provenance.source_turn_ids.filter((sourceId) => turnsById.has(sourceId))
                : [];
        },
        (from, to) => {
            const located = turnsById.get(from);
            add(
                'DERIVED_PROVENANCE_CYCLE',
                `${located?.path ?? ''}/provenance/source_turn_ids`,
                `Derived-turn provenance contains a cycle through ${diagnosticValue(to)}`,
                from,
            );
        },
    );

    for (const [assetId, asset] of Object.entries(document.assets)) {
        const path = recordPath('assets', assetId);
        if (asset.provenance.type === 'generated' && !hasOwn(document.generations, asset.provenance.generation_id)) {
            add(
                'REFERENCE_NOT_FOUND',
                `${path}/provenance/generation_id`,
                `Generation ${diagnosticValue(asset.provenance.generation_id)} does not exist`,
                asset.id,
            );
        }
        if (asset.provenance.type === 'derived') {
            if (
                asset.provenance.source_asset_id === asset.id ||
                !hasOwn(document.assets, asset.provenance.source_asset_id)
            ) {
                add(
                    'REFERENCE_NOT_FOUND',
                    `${path}/provenance/source_asset_id`,
                    `Derived asset ${diagnosticValue(asset.provenance.source_asset_id)} is missing or self-referential`,
                    asset.id,
                );
            }
        }
        if (asset.provenance.type === 'received' && asset.provenance.source_turn_id !== undefined) {
            if (!turnsById.has(asset.provenance.source_turn_id)) {
                add(
                    'REFERENCE_NOT_FOUND',
                    `${path}/provenance/source_turn_id`,
                    `Source turn ${diagnosticValue(asset.provenance.source_turn_id)} does not exist`,
                    asset.id,
                );
            }
        }
    }

    validateDirectedCycles(
        Object.keys(document.assets),
        (assetId) => {
            const asset = hasOwn(document.assets, assetId) ? document.assets[assetId] : undefined;
            return asset?.provenance.type === 'derived' && hasOwn(document.assets, asset.provenance.source_asset_id)
                ? [asset.provenance.source_asset_id]
                : [];
        },
        (from, to) => {
            add(
                'DERIVED_PROVENANCE_CYCLE',
                `${recordPath('assets', from)}/provenance/source_asset_id`,
                `Derived-asset provenance contains a cycle through ${diagnosticValue(to)}`,
                from,
            );
        },
    );

    const terminalReceiptCalls = new Map<string, string>();
    for (const [receiptId, receipt] of Object.entries(document.execution_receipts)) {
        const path = recordPath('execution_receipts', receiptId);
        const call = callsById.get(receipt.call_id);
        // Receipts outlive logically deleted call/result content. Absence is therefore valid; when
        // the content remains present, its executor and result identity are checked.
        if (call !== undefined && call.block.executor !== receipt.executor) {
            add('TOOL_RECEIPT_MISMATCH', `${path}/executor`, 'Execution receipt executor does not match its tool call');
        }
        const prior = terminalReceiptCalls.get(receipt.call_id);
        if (prior !== undefined) {
            add(
                'DUPLICATE_TERMINAL_RESULT',
                `${path}/call_id`,
                `Call ${diagnosticValue(receipt.call_id)} already has terminal receipt ${diagnosticValue(prior)}`,
                receipt.call_id,
            );
        } else {
            terminalReceiptCalls.set(receipt.call_id, receipt.id);
        }
        if (receipt.result_turn_id !== undefined) {
            const resultTurn = turnsById.get(receipt.result_turn_id);
            if (resultTurn !== undefined) {
                const matchingResult = resultTurn.turn.blocks.find(
                    (block): block is ToolResultBlock =>
                        block.type === 'tool_result' && block.call_id === receipt.call_id,
                );
                if (matchingResult === undefined) {
                    add(
                        'TOOL_RESULT_UNRESOLVED',
                        `${path}/result_turn_id`,
                        'Execution receipt result turn does not contain its call result',
                        receipt.id,
                    );
                } else if (matchingResult.status !== receipt.status) {
                    add(
                        'TOOL_RECEIPT_MISMATCH',
                        `${path}/status`,
                        'Execution receipt status does not match its retained terminal result',
                        receipt.id,
                    );
                }
            }
        }
    }

    for (const located of blocksById.values()) {
        const { block, path } = located;
        if (block.type === 'tool_call' && block.definition_id !== undefined) {
            const definition = hasOwn(document.tool_definitions, block.definition_id)
                ? document.tool_definitions[block.definition_id]
                : undefined;
            if (definition === undefined) {
                add(
                    'REFERENCE_NOT_FOUND',
                    `${path}/definition_id`,
                    `Tool definition ${diagnosticValue(block.definition_id)} does not exist`,
                    block.id,
                );
            } else if (definition.name !== block.tool_name) {
                add(
                    'TOOL_DEFINITION_MISMATCH',
                    `${path}/tool_name`,
                    `Tool call name ${diagnosticValue(block.tool_name)} does not match its pinned definition`,
                    block.id,
                );
            }
        }
        if (block.type === 'tool_result' && !callsById.has(block.call_id) && !terminalReceiptCalls.has(block.call_id)) {
            add(
                'TOOL_RESULT_UNRESOLVED',
                `${path}/call_id`,
                `Tool result call ${diagnosticValue(block.call_id)} has no retained call or terminal receipt`,
                block.id,
            );
        }
        if (
            block.type === 'image' ||
            block.type === 'document' ||
            block.type === 'audio' ||
            block.type === 'video' ||
            block.type === 'external_reference'
        ) {
            const asset = hasOwn(document.assets, block.asset_id) ? document.assets[block.asset_id] : undefined;
            if (asset === undefined) {
                add(
                    'REFERENCE_NOT_FOUND',
                    `${path}/asset_id`,
                    `Asset ${diagnosticValue(block.asset_id)} does not exist`,
                    block.id,
                );
            } else if (block.type !== 'external_reference') {
                if (asset.kind !== block.type) {
                    add(
                        'ASSET_KIND_MISMATCH',
                        `${path}/asset_id`,
                        `${block.type} block references ${asset.kind} asset ${diagnosticValue(asset.id)}`,
                        block.id,
                    );
                }
                validateMediaSelection(block, path, asset, add);
            }
            if (
                block.type === 'external_reference' &&
                block.retrieval.tool_definition_id !== undefined &&
                !hasOwn(document.tool_definitions, block.retrieval.tool_definition_id)
            ) {
                add(
                    'REFERENCE_NOT_FOUND',
                    `${path}/retrieval/tool_definition_id`,
                    `Retrieval tool definition ${diagnosticValue(block.retrieval.tool_definition_id)} does not exist`,
                    block.id,
                );
            }
        }
        if (block.type === 'native_replay') {
            validateReplayDependencies(block, path, turnsById, blocksById, callsById, requestIds, add);
        }
    }

    for (const [operationId, receipt] of Object.entries(document.operation_receipts)) {
        const path = recordPath('operation_receipts', operationId);
        if (receipt.conversation_id !== document.id) {
            add('REFERENCE_NOT_FOUND', `${path}/conversation_id`, 'Operation receipt belongs to another conversation');
        }
        if (receipt.base_revision > receipt.result_revision || receipt.result_revision > document.revision) {
            add(
                'CONTEXT_REVISION_INVALID',
                path,
                'Operation receipt revisions are reversed or newer than the document',
            );
        }
    }

    for (const [compactionId, compaction] of Object.entries(document.compactions)) {
        validateCompaction(compaction, recordPath('compactions', compactionId), turnsById, blocksById, document, add);
    }

    for (const [compactionId, compaction] of Object.entries(document.compactions)) {
        const supersededId = compaction.supersedes_compaction_id;
        if (supersededId !== undefined && !hasOwn(document.compactions, supersededId)) {
            add(
                'REFERENCE_NOT_FOUND',
                `${recordPath('compactions', compactionId)}/supersedes_compaction_id`,
                `Superseded compaction ${diagnosticValue(supersededId)} does not exist`,
                compactionId,
            );
        }
    }
    validateDirectedCycles(
        Object.keys(document.compactions),
        (compactionId) => {
            const nextId = document.compactions[compactionId]?.supersedes_compaction_id;
            return nextId !== undefined && hasOwn(document.compactions, nextId) ? [nextId] : [];
        },
        (from, to) => {
            add(
                'SUPERSEDES_CYCLE',
                `${recordPath('compactions', from)}/supersedes_compaction_id`,
                `Compaction supersedes chain contains a cycle through ${diagnosticValue(to)}`,
                from,
            );
        },
    );

    const contextEntryIds = new Map<string, string>();
    const activeSelections = new Map<string, ActiveSelectionIndex>();
    const sourceSelectionsForEntry = (entry: ContextEntry, path: string, located: LocatedTurn): ActiveSelection[] => {
        if (entry.type === 'source_turn') {
            return [
                {
                    turn_id: entry.turn_id,
                    all_blocks: entry.block_ids === undefined,
                    block_ids: new Set(entry.block_ids ?? []),
                    origin: 'direct',
                    path,
                },
            ];
        }
        const provenance = located.turn.provenance;
        if (provenance.type !== 'derived') {
            return [];
        }
        if (provenance.source_block_ids === undefined) {
            return provenance.source_turn_ids.map((turnId) => ({
                turn_id: turnId,
                all_blocks: true,
                block_ids: new Set<string>(),
                origin: 'replacement',
                path,
                compaction_id: entry.compaction_id,
            }));
        }
        const blocksByTurn = new Map<string, Set<string>>();
        for (const blockId of provenance.source_block_ids) {
            const sourceTurnId = blocksById.get(blockId)?.turn_id;
            if (sourceTurnId === undefined) {
                continue;
            }
            const selectedBlocks = blocksByTurn.get(sourceTurnId) ?? new Set<string>();
            selectedBlocks.add(blockId);
            blocksByTurn.set(sourceTurnId, selectedBlocks);
        }
        return [...blocksByTurn].map(([turnId, blockIds]) => ({
            turn_id: turnId,
            all_blocks: false,
            block_ids: blockIds,
            origin: 'replacement',
            path,
            compaction_id: entry.compaction_id,
        }));
    };
    const registerActiveSelection = (selection: ActiveSelection, entryId: string): void => {
        const existing = activeSelections.get(selection.turn_id);
        let overlap: ActiveSelection | undefined;
        if (existing !== undefined) {
            if (selection.all_blocks) {
                overlap = existing.first;
            } else if (existing.all !== undefined) {
                overlap = existing.all;
            } else {
                for (const blockId of selection.block_ids) {
                    const prior = existing.by_block_id.get(blockId);
                    if (prior !== undefined) {
                        overlap = prior;
                        break;
                    }
                }
            }
        }
        if (overlap !== undefined && selectionsOverlap(overlap, selection)) {
            const directAndReplacement = overlap.origin !== selection.origin;
            add(
                directAndReplacement ? 'CONTEXT_DIRECT_REPLACEMENT_OVERLAP' : 'CONTEXT_SELECTION_OVERLAP',
                selection.path,
                `Context selection overlaps the active selection at ${overlap.path}`,
                entryId,
            );
        }
        const index = existing ?? { first: selection, by_block_id: new Map<string, ActiveSelection>() };
        if (selection.all_blocks && index.all === undefined) {
            index.all = selection;
        }
        for (const blockId of selection.block_ids) {
            if (!index.by_block_id.has(blockId)) {
                index.by_block_id.set(blockId, selection);
            }
        }
        activeSelections.set(selection.turn_id, index);
    };
    for (let index = 0; index < document.context.entries.length; index += 1) {
        const entry = document.context.entries[index];
        const path = `/context/entries/${index}`;
        registerId(entry.id, 'context entry', `${path}/id`);
        const prior = contextEntryIds.get(entry.id);
        if (prior !== undefined) {
            add(
                'DUPLICATE_ID',
                `${path}/id`,
                `Context entry ${diagnosticValue(entry.id)} duplicates ${prior}`,
                entry.id,
            );
        } else {
            contextEntryIds.set(entry.id, path);
        }
        const located = turnsById.get(entry.turn_id);
        const validTarget =
            entry.type === 'source_turn'
                ? located?.source === true
                : located?.source === false && located.compaction_id === entry.compaction_id;
        if (!validTarget || located === undefined) {
            add(
                'REFERENCE_NOT_FOUND',
                `${path}/turn_id`,
                `Context turn ${diagnosticValue(entry.turn_id)} is missing or has the wrong source kind`,
            );
            continue;
        }
        if (entry.block_ids !== undefined) {
            let priorBlockIndex = -1;
            const turnBlockIds = located.turn.blocks.map((block) => block.id);
            for (let blockIndex = 0; blockIndex < entry.block_ids.length; blockIndex += 1) {
                const selectedId = entry.block_ids[blockIndex];
                const currentBlockIndex = turnBlockIds.indexOf(selectedId);
                if (currentBlockIndex <= priorBlockIndex) {
                    add(
                        'CONTEXT_BLOCK_ORDER_INVALID',
                        `${path}/block_ids/${blockIndex}`,
                        `Selected block ${diagnosticValue(selectedId)} is missing, duplicated, or out of source order`,
                        entry.id,
                    );
                }
                priorBlockIndex = currentBlockIndex;
            }
        }
        for (const selection of sourceSelectionsForEntry(entry, path, located)) {
            registerActiveSelection(selection, entry.id);
        }
    }
    for (let index = 0; index < document.context.active_tool_definition_ids.length; index += 1) {
        const definitionId = document.context.active_tool_definition_ids[index];
        if (!hasOwn(document.tool_definitions, definitionId)) {
            add(
                'REFERENCE_NOT_FOUND',
                `/context/active_tool_definition_ids/${index}`,
                `Active tool definition ${diagnosticValue(definitionId)} does not exist`,
            );
        }
    }
    for (let index = 0; index < document.context.protected_entry_ids.length; index += 1) {
        const entryId = document.context.protected_entry_ids[index];
        if (!contextEntryIds.has(entryId)) {
            add(
                'REFERENCE_NOT_FOUND',
                `/context/protected_entry_ids/${index}`,
                `Protected context entry ${diagnosticValue(entryId)} does not exist`,
            );
        }
    }
    if (
        document.context.cache_intent?.stable_through_entry_id !== undefined &&
        !contextEntryIds.has(document.context.cache_intent.stable_through_entry_id)
    ) {
        add(
            'REFERENCE_NOT_FOUND',
            '/context/cache_intent/stable_through_entry_id',
            `Cache boundary ${diagnosticValue(document.context.cache_intent.stable_through_entry_id)} does not exist`,
        );
    }
    const retrievalRequirementIds = new Set<string>();
    for (let index = 0; index < document.context.retrieval_requirements.length; index += 1) {
        const requirement = document.context.retrieval_requirements[index];
        const path = `/context/retrieval_requirements/${index}`;
        registerId(requirement.id, 'retrieval requirement', `${path}/id`);
        if (retrievalRequirementIds.has(requirement.id)) {
            add(
                'DUPLICATE_ID',
                `${path}/id`,
                `Retrieval requirement ${diagnosticValue(requirement.id)} is duplicated`,
                requirement.id,
            );
        }
        retrievalRequirementIds.add(requirement.id);
        if (!hasOwn(document.assets, requirement.asset_id)) {
            add(
                'REFERENCE_NOT_FOUND',
                `${path}/asset_id`,
                `Retrieval asset ${diagnosticValue(requirement.asset_id)} does not exist`,
            );
        }
        if (
            requirement.retrieval.tool_definition_id !== undefined &&
            !hasOwn(document.tool_definitions, requirement.retrieval.tool_definition_id)
        ) {
            add(
                'REFERENCE_NOT_FOUND',
                `${path}/retrieval/tool_definition_id`,
                `Retrieval tool definition ${diagnosticValue(requirement.retrieval.tool_definition_id)} does not exist`,
            );
        }
    }

    const processorKeys = new Set<string>();
    for (let index = 0; index < document.processing.processors.length; index += 1) {
        const processor = document.processing.processors[index];
        const key = `${processor.id}\u0000${processor.version}`;
        if (processorKeys.has(key)) {
            add(
                'DUPLICATE_ID',
                `/processing/processors/${index}`,
                `Processor ${diagnosticValue(processor.id)}@${diagnosticValue(processor.version)} is duplicated`,
            );
        }
        processorKeys.add(key);
    }

    return diagnostics;
}
