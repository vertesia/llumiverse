import { createConversationDocument, createUserTurn } from './builders.js';
import { deriveConversationId } from './identity.js';
import type { ConversationDocument } from './types.js';
import { parseConversationDocument } from './validation.js';

export async function createCheckpointSummaryFork(
    source: ConversationDocument,
    operationId: string,
): Promise<ConversationDocument> {
    const forkId = await deriveConversationId('checkpoint_summary', source.id, operationId);
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
    const selectedResultIds = new Set(
        selected.flatMap(({ turn, blocks }) =>
            turn.kind === 'tool'
                ? blocks.filter((block) => block.type === 'tool_result').map((block) => block.call_id)
                : [],
        ),
    );
    const transcript = selected.flatMap(({ turn, blocks }) => {
        const visibleBlocks = blocks.filter((block) => block.type !== 'native_replay');
        if (visibleBlocks.length === 0) return [];
        const rendered = visibleBlocks.flatMap((block) => renderCheckpointBlock(block, selectedResultIds));
        return rendered.length > 0 ? [`[${turn.kind}:${turn.id}]`, ...rendered] : [];
    });
    const fork = createConversationDocument({ id: forkId, created_at: source.updated_at });
    fork.revision = source.revision;
    fork.context.revision = source.context.revision;
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
