import { describe, expect, it } from 'vitest';
import { isConversationDocumentFormat, renderContentBlockText, renderConversationText } from '../src/index.js';
import { emptyDocument, textBlock, toolResultTurn, userTurn } from './fixtures.js';

describe('human-readable canonical history', () => {
    it('renders source history independently of active context and nested media', () => {
        const document = emptyDocument();
        const user = userTurn('user');
        user.blocks = [textBlock('question', 'Look here'), { type: 'image', id: 'image', asset_id: 'asset' }];
        const result = toolResultTurn('result', 'call');
        result.blocks[0].content = [textBlock('answer', 'Found it'), { type: 'audio', id: 'audio', asset_id: 'sound' }];
        document.turns = [user, result];
        expect(renderConversationText(document)).toBe(
            '[USER]: Look here [Image]\n\n[TOOL]: [TOOL RESULT]: call → Found it [Audio]',
        );
    });

    it('does not expose opaque replay or extension payloads', () => {
        expect(
            renderContentBlockText({
                id: 'extension',
                type: 'extension',
                namespace: 'example',
                version: '1',
                payload: { secret: 'opaque' },
            }),
        ).toBe('');
    });

    it('recognizes only an own data-property envelope without executing getters', () => {
        expect(isConversationDocumentFormat(emptyDocument())).toBe(true);
        expect(isConversationDocumentFormat({ format: 'other' })).toBe(false);
        expect(isConversationDocumentFormat(Object.create({ format: 'llumiverse.conversation' }))).toBe(false);
        expect(
            isConversationDocumentFormat({
                get format() {
                    throw new Error('must not execute');
                },
            }),
        ).toBe(false);
        // Envelope recognition is intentionally not schema validation.
        expect(isConversationDocumentFormat({ format: 'llumiverse.conversation' })).toBe(true);
    });
});
