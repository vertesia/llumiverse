import type { ConversationDocument } from '@llumiverse/conversation';

export function getConversationId(document: ConversationDocument): string {
    return document.id;
}
