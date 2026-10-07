import type {
    ConversationTurn,
    IndexedConversationSelectedContext,
    RequestSourceWorkingSet,
} from '@llumiverse/conversation';
import { ConversationTurnSchema } from '@llumiverse/conversation/schemas';

/** Dependency projection only; never a complete ConversationDocument or readiness authority. */
export function selectedWorkingSetSource(
    workingSet: Pick<RequestSourceWorkingSet, 'turns' | 'context' | 'assets'> & {
        replacement_turns?: RequestSourceWorkingSet['replacement_turns'];
        indexed_reference_evidence?: IndexedConversationSelectedContext;
    },
) {
    const materialize = (projection: RequestSourceWorkingSet['turns'][number]): ConversationTurn =>
        ConversationTurnSchema.parse({ ...projection.header, blocks: projection.selected_blocks });
    const turns = workingSet.turns.map(materialize);
    const grouped = new Map<string, ConversationTurn[]>();
    for (const { compaction_id, projection } of workingSet.replacement_turns ?? []) {
        const prior = grouped.get(compaction_id) ?? [];
        prior.push(materialize(projection));
        grouped.set(compaction_id, prior);
    }
    const compactions = Object.fromEntries([...grouped].map(([id, replacement_turns]) => [id, { replacement_turns }]));
    return {
        turns,
        context: workingSet.context,
        compactions,
        assets: workingSet.assets,
        ...(workingSet.indexed_reference_evidence === undefined
            ? {}
            : { indexed_reference_evidence: workingSet.indexed_reference_evidence }),
    };
}
