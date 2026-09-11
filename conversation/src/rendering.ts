import type { ContentBlock, ConversationDocument } from './types.js';

/** Human-readable projection; never hydrates assets or exposes opaque replay bytes. */
export function renderContentBlockText(block: ContentBlock): string {
    switch (block.type) {
        case 'text':
        case 'reasoning':
            return block.text;
        case 'json':
            return JSON.stringify(block.value);
        case 'image':
            return '[Image]';
        case 'document':
            return '[Document]';
        case 'audio':
            return '[Audio]';
        case 'video':
            return '[Video]';
        case 'tool_call':
            return `[TOOL CALL]: ${block.tool_name}(...)`;
        case 'tool_result':
            return `[TOOL RESULT]: ${block.call_id} → ${block.content.map(renderContentBlockText).filter(Boolean).join(' ')}`;
        case 'external_reference':
            return block.preview ?? `[External content: ${block.description}]`;
        case 'native_replay':
        case 'extension':
            return '';
    }
}

/** Renders source history, preserving order independently of the current model-context selection. */
export function renderConversationText(document: ConversationDocument): string {
    const lines: string[] = [];
    for (const turn of document.turns) {
        const text = turn.blocks.map(renderContentBlockText).filter(Boolean).join(' ');
        if (!text) continue;
        const role = turn.kind === 'agent' ? 'ASSISTANT' : turn.kind.toUpperCase();
        lines.push(`[${role}]: ${text}`);
    }
    return lines.join('\n\n');
}
