import { preflightNativeConversationImportInput } from '@llumiverse/conversation';
import { describe, expect, it, vi } from 'vitest';
import { bedrockConverseJsonValue } from '../bedrock/bedrock-converse-conversation-adapter.js';
import { importBedrockConverseHistory } from './index.js';

const origin = {
    conversation_id: 'byte-source',
    recorded_at: '2026-10-01T00:00:00.000Z',
    provider: 'recorded-bedrock',
};
function history(bytes: Uint8Array) {
    return { messages: [{ role: 'user', content: [{ image: { format: 'png', source: { bytes } } }] }] };
}

describe('native byte-view import safety', () => {
    it.each(['length', 'buffer', 'byteOffset', 'byteLength'])(
        'rejects own %s accessors before conversion',
        async (key) => {
            const bytes = new Uint8Array([97, 98, 99]);
            let accessed = false;
            Object.defineProperty(bytes, key, {
                get() {
                    accessed = true;
                    return 0;
                },
            });
            const checked = preflightNativeConversationImportInput(history(bytes));
            await expect(importBedrockConverseHistory(history(bytes), origin)).rejects.toMatchObject({
                code: 'IMPORT_INVALID_HISTORY',
            });
            expect(checked.success).toBe(false);
            expect(accessed).toBe(false);
        },
    );

    it.each([true, false])('excludes own annotations explicitly when enumerable=%s', async (enumerable) => {
        const bytes = new Uint8Array([97, 98, 99]);
        Object.defineProperty(bytes, 'native_annotation', { value: 'not protocol byte data', enumerable });
        const result = await importBedrockConverseHistory(history(bytes), origin);
        expect(result.report.diagnostics.map((diagnostic) => diagnostic.code)).toContain(
            'IMPORT_BYTE_VIEW_PROPERTIES_EXCLUDED',
        );
        expect(JSON.stringify(result.document)).not.toContain('not protocol byte data');
        expect(Object.values(result.document.assets)[0]).toMatchObject({
            storage: { type: 'inline_base64', data: 'YWJj' },
        });
    });

    it('does not enumerate byte keys/descriptors or invoke byte annotation hooks during preflight/encoding', async () => {
        const bytes = Buffer.from([97, 98, 99]);
        let accessed = false;
        Object.defineProperty(bytes, 'annotation', {
            get() {
                accessed = true;
                throw new Error('must not invoke');
            },
        });
        Object.defineProperty(bytes, Symbol.iterator, {
            get() {
                accessed = true;
                throw new Error('must not invoke');
            },
        });
        const ownKeys = Reflect.ownKeys;
        const descriptors = Object.getOwnPropertyDescriptors;
        const keysSpy = vi.spyOn(Reflect, 'ownKeys').mockImplementation((value) => {
            if (value instanceof Uint8Array) throw new Error('byte key enumeration forbidden');
            return ownKeys(value);
        });
        const descriptorsSpy = vi.spyOn(Object, 'getOwnPropertyDescriptors').mockImplementation((value) => {
            if (value instanceof Uint8Array) throw new Error('byte descriptor enumeration forbidden');
            return descriptors(value);
        });
        try {
            const result = await importBedrockConverseHistory(history(bytes), origin);
            expect(Object.values(result.document.assets)[0]).toMatchObject({
                storage: { type: 'inline_base64', data: 'YWJj' },
            });
            expect(bedrockConverseJsonValue(bytes)).toEqual({ _llumiverse_bedrock_bytes: 'YWJj' });
            expect(accessed).toBe(false);
        } finally {
            keysSpy.mockRestore();
            descriptorsSpy.mockRestore();
        }
    });

    it('accepts undecorated Node Buffer bytes without changing the view or payload', async () => {
        const bytes = Buffer.from([97, 98, 99]);
        const result = await importBedrockConverseHistory(history(bytes), origin);
        expect(Object.values(result.document.assets)[0]).toMatchObject({
            storage: { type: 'inline_base64', data: 'YWJj' },
        });
        expect([...bytes]).toEqual([97, 98, 99]);
    });

    it('preserves a multi-megabyte plain byte view without descriptor/child expansion', async () => {
        const bytes = new Uint8Array(4 * 1024 * 1024).fill(97);
        const result = await importBedrockConverseHistory(history(bytes), origin);
        const asset = Object.values(result.document.assets)[0];
        expect(asset.storage.type).toBe('inline_base64');
        if (asset.storage.type !== 'inline_base64') throw new Error('Expected byte-preserving asset');
        expect(Buffer.from(asset.storage.data, 'base64').equals(Buffer.from(bytes))).toBe(true);
    }, 20_000);
});
