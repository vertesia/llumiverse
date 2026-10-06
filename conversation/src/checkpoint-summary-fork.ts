import { createConversationDocument, createUserTurn } from './builders.js';
import { deriveConversationId } from './identity.js';
import { preflightJsonInput } from './json-preflight.js';
import type {
    IndexedConversationRoot,
    IndexedConversationSelectedContext,
    IndexedProcessingSelectedContext,
} from './schemas/indexed-head.js';
import type { ConversationDocument } from './types.js';
import { parseConversationDocument } from './validation.js';

export async function createCheckpointSummaryFork(
    source: ConversationDocument,
    operationId: string,
): Promise<ConversationDocument> {
    const sourceTurns = new Map(source.turns.map((turn) => [turn.id, turn]));
    const selected = source.context.entries.flatMap((entry) => {
        const turn =
            entry.type === 'source_turn'
                ? sourceTurns.get(entry.turn_id)
                : source.compactions[entry.compaction_id]?.replacement_turns.find(
                      (candidate) => candidate.id === entry.turn_id,
                  );
        if (!turn) throw new Error(`Checkpoint context references missing turn ${entry.turn_id}`);
        const blockIds = entry.block_ids ? new Set(entry.block_ids) : undefined;
        if (turn.model_visibility === 'exclude') return [];
        return [
            {
                turn,
                blocks: turn.blocks.filter((block) => !blockIds || blockIds.has(block.id)),
            },
        ];
    });
    return createSelectedCheckpointSummaryFork(
        {
            id: source.id,
            revision: source.revision,
            updated_at: source.updated_at,
            context_revision: source.context.revision,
        },
        selected,
        operationId,
    );
}

/** Exact selected projections supply original positions/accepted dependencies. This constructs
 * a genuine separate fork, never a partial source document or new source authority. */
export async function createCheckpointSummaryForkFromIndexedSelection(
    root: IndexedConversationRoot,
    selection: IndexedConversationSelectedContext | IndexedProcessingSelectedContext,
    operationId: string,
): Promise<ConversationDocument> {
    if (
        selection.source.conversation_id !== root.source.conversation_id ||
        selection.source.revision !== root.source.revision
    )
        throw new TypeError('Checkpoint selected context differs from its exact indexed source');
    const turns = new Map(selection.turns.map((turn) => [turn.header.id, turn]));
    const replacements = new Map(
        (selection.replacement_turns ?? []).map((turn) => [
            JSON.stringify([turn.compaction_id, turn.projection.header.id]),
            turn.projection,
        ]),
    );
    const selected = selection.context.entries.flatMap((entry) => {
        const projection =
            entry.type === 'source_turn'
                ? turns.get(entry.turn_id)
                : replacements.get(JSON.stringify([entry.compaction_id, entry.turn_id]));
        if (!projection) throw new Error(`Checkpoint selected context lacks turn ${entry.turn_id}`);
        if (projection.header.model_visibility === 'exclude') return [];
        const ids = entry.block_ids ? new Set(entry.block_ids) : undefined;
        const blocks = projection.selected_blocks.filter((block) => !ids || ids.has(block.id));
        if (ids && blocks.length !== ids.size)
            throw new Error('Checkpoint selected context lacks an exact active block');
        return [{ turn: projection.header, blocks }];
    });
    return createSelectedCheckpointSummaryFork(
        {
            id: root.source.conversation_id,
            revision: root.source.revision,
            updated_at: root.updated_at,
            context_revision: selection.context.revision,
        },
        selected,
        operationId,
    );
}

export class CheckpointSummaryForkCapacityError extends RangeError {
    constructor(message: string) {
        super(message);
        this.name = 'CheckpointSummaryForkCapacityError';
    }
}

// Reserve envelope/fork lineage overhead inside the 16MiB private JSON response budget.
const MAX_CHECKPOINT_TRANSCRIPT_JSON_BYTES = 16 * 1024 * 1024 - 64 * 1024;
type SelectedCheckpointTurn = {
    turn: Pick<ConversationDocument['turns'][number], 'id' | 'kind'>;
    blocks: readonly ConversationDocument['turns'][number]['blocks'][number][];
};
async function createSelectedCheckpointSummaryFork(
    source: { id: string; revision: number; updated_at: string; context_revision: number },
    selected: readonly SelectedCheckpointTurn[],
    operationId: string,
): Promise<ConversationDocument> {
    const forkId = await deriveConversationId('checkpoint_summary', source.id, operationId);
    const selectedResultIds = new Set(
        selected.flatMap(({ turn, blocks }) =>
            turn.kind === 'tool'
                ? blocks.filter((block) => block.type === 'tool_result').map((block) => block.call_id)
                : [],
        ),
    );
    const transcript: string[] = [];
    let encodedBytes = 2;
    const append = (line: string) => {
        const remaining = MAX_CHECKPOINT_TRANSCRIPT_JSON_BYTES - encodedBytes;
        const inspected = preflightJsonInput(line, { max_bytes: Math.max(1, remaining) });
        if (!inspected.success)
            throw new CheckpointSummaryForkCapacityError('Checkpoint fork transcript exceeds its encoded JSON budget');
        encodedBytes += inspected.bytes - 2 + (transcript.length ? 2 : 0);
        if (encodedBytes > MAX_CHECKPOINT_TRANSCRIPT_JSON_BYTES)
            throw new CheckpointSummaryForkCapacityError('Checkpoint fork transcript exceeds its encoded JSON budget');
        transcript.push(line);
    };
    for (const { turn, blocks } of selected) {
        const visibleBlocks = blocks.filter((block) => block.type !== 'native_replay');
        if (visibleBlocks.length === 0) continue;
        const rendered = visibleBlocks.flatMap((block) => renderCheckpointBlock(block, selectedResultIds));
        if (rendered.length > 0) {
            append(`[${turn.kind}:${turn.id}]`);
            for (const line of rendered) append(line);
        }
    }
    const fork = createConversationDocument({ id: forkId, created_at: source.updated_at });
    fork.revision = source.revision;
    fork.context.revision = source.context_revision;
    fork.lineage = {
        parents: [{ relation: 'fork', source: { conversation_id: source.id, revision: source.revision } }],
    };
    const transcriptTurn = createUserTurn({
        id: await deriveConversationId('turn', forkId, 'source-transcript'),
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: source.updated_at, completed_at: source.updated_at },
        provenance: { type: 'received' },
        model_visibility: 'include',
        blocks: [
            {
                id: await deriveConversationId('block', forkId, 'source-transcript'),
                type: 'text',
                text: transcript.join('\n'),
                format: 'plain',
            },
        ],
    });
    fork.turns = [transcriptTurn];
    fork.context.entries = [{ id: 'summary-source-context', type: 'source_turn', turn_id: transcriptTurn.id }];
    return parseConversationDocument(fork);
}

function renderCheckpointBlock(
    block: ConversationDocument['turns'][number]['blocks'][number],
    selectedResultIds: ReadonlySet<string>,
): string[] {
    if (block.type === 'text') return [block.text];
    if (block.type === 'json') return [JSON.stringify(block.value)];
    if (block.type === 'reasoning') return [`[reasoning summary] ${block.text}`];
    if (block.type === 'tool_call') {
        const state = selectedResultIds.has(block.call_id) ? 'answered' : 'pending';
        return [`[tool call ${block.call_id}: ${state}] ${block.tool_name} ${JSON.stringify(block.arguments)}`];
    }
    if (block.type === 'tool_result') {
        return [
            `[tool result ${block.call_id}: ${block.status}]`,
            ...block.content.flatMap((content) => renderCheckpointBlock(content, selectedResultIds)),
        ];
    }
    if (block.type === 'image' || block.type === 'audio' || block.type === 'video' || block.type === 'document') {
        return [`[${block.type} asset ${block.asset_id}]`];
    }
    if (block.type === 'external_reference') return [`[external reference asset ${block.asset_id}]`];
    return [];
}
