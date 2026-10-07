import { describe, expect, it } from 'vitest';
import {
    appendConversationRecords,
    createConversationDocument,
    fingerprintJson,
    hashContentBytes,
    renderCanonicalToolRetrievalResult,
} from '../src/index.js';

const at = '2026-01-01T00:00:00.000Z';

async function render(content: string, range: { start_line: number; end_line?: number; line_numbers?: boolean }) {
    const bytes = new TextEncoder().encode(content);
    const asset = {
        id: 'asset:one',
        kind: 'text' as const,
        mime_type: 'text/plain',
        ...(await hashContentBytes(bytes)),
        storage: {
            type: 'external' as const,
            resolver: 'vertesia.agent_artifact',
            locator: { storage_id: 'subject', artifact_path: 'archive/one.txt' },
        },
        provenance: { type: 'received' as const },
        created_at: at,
    };
    const retrieval = {
        capability: 'read_artifact',
        version: 1,
        tool_definition_id: 'definition:read',
        arguments: { path: 'archive/one.txt', asset_id: asset.id },
    };
    const document = appendConversationRecords(
        createConversationDocument({ id: 'conversation', created_at: at }),
        {
            assets: [asset],
            tool_definitions: [
                { id: 'definition:read', name: 'read_artifact', version: 'content-hash', input_schema: true },
            ],
            active_tool_definition_ids: ['definition:read'],
        },
        {
            operation_id: 'asset:accepted',
            expected_revision: 0,
            recorded_at: at,
            payload_fingerprint: await fingerprintJson(asset),
        },
    ).document;
    const call = {
        id: 'call:block',
        type: 'tool_call' as const,
        call_id: 'call',
        tool_name: 'read_artifact',
        executor: 'application' as const,
        definition_id: 'definition:read',
        arguments: {
            type: 'json' as const,
            value: { ...retrieval.arguments, ...range },
        },
    };
    document.turns.push(
        {
            id: 'call:turn',
            kind: 'agent',
            status: 'completed',
            authority: 'ordinary',
            model_visibility: 'include',
            provenance: { type: 'imported', source: 'retrieval renderer fixture' },
            timestamps: { recorded_at: at },
            blocks: [call],
        },
        {
            id: 'ref:turn',
            kind: 'user',
            status: 'completed',
            authority: 'ordinary',
            model_visibility: 'include',
            provenance: { type: 'received' },
            timestamps: { recorded_at: at },
            blocks: [
                {
                    id: 'ref:block',
                    type: 'external_reference',
                    asset_id: asset.id,
                    original_type: 'text',
                    content_hash: asset.content_hash,
                    description: 'Archived text',
                    retrieval,
                },
            ],
        },
    );
    document.context.entries.push(
        { id: 'call:entry', type: 'source_turn', turn_id: 'call:turn' },
        { id: 'ref:entry', type: 'source_turn', turn_id: 'ref:turn' },
    );
    document.context.retrieval_requirements.push({
        id: 'requirement',
        asset_id: asset.id,
        retrieval,
        accepted_asset_operation_id: 'asset:accepted',
    });
    return renderCanonicalToolRetrievalResult(
        document,
        {
            conversation: { conversation_id: document.id, revision: document.revision },
            turn_id: 'call:turn',
            block_id: call.id,
            call_id: call.call_id,
            call_fingerprint: await fingerprintJson(call),
        },
        asset.id,
        bytes,
    );
}

describe('canonical retrieval line projection', () => {
    it('retains ordinary line formatting and advances by line at a whole-line boundary', async () => {
        const lines = ['alpha', 'beta', 'gamma'];
        const result = await render(lines.join('\n'), { start_line: 1, end_line: 2, line_numbers: true });
        const content = JSON.parse(result.text).content as string;
        expect(content).toContain('1\talpha\n   2\tbeta');
        expect(content).toContain('Use start_line=3 to continue.');
        expect(result.projection).toMatchObject({ start_line: 1, end_line: 2, line_numbers: true });
    });

    it('uses the exact UTF-8 byte offset for a truncated line after CRLF and multibyte text', async () => {
        const prefix = 'leadé\r\n';
        const line = `${'a'.repeat(4999)}🙂remaining`;
        const result = await render(`${prefix}${line}\r\nlast`, { start_line: 2, end_line: 2 });
        const content = JSON.parse(result.text).content as string;
        const offset = new TextEncoder().encode(prefix).byteLength + 4999;
        expect(content).toContain('a'.repeat(4999));
        expect(content).not.toContain('remaining');
        expect(content).toContain(`Use start_byte=${offset} to continue this line.`);
        expect(content).not.toContain('Use start_line=3');
        expect(result.projection).toMatchObject({ start_line: 2, end_line: 2 });
    });

    it('bounds a large selected range by its first complete lines', async () => {
        const lines = Array.from(
            { length: 2000 },
            (_, index) => `${String(index).padStart(4, '0')}${'x'.repeat(2496)}`,
        );
        const result = await render(lines.join('\n'), { start_line: 1, end_line: 2000 });
        const content = JSON.parse(result.text).content as string;
        expect(content).toContain(lines[0]);
        expect(content).not.toContain(lines[1]);
        expect(content).toContain('Use start_line=2 to continue.');
        expect(result.projection).toMatchObject({ start_line: 1, end_line: 1 });
    });
});
