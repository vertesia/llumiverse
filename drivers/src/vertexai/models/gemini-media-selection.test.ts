import { GoogleGenAI } from '@google/genai';
import { describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { geminiMediaSelectionFields } from './gemini-media-selection.js';

const DOCX = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
const XLSX = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';

function fields(mime_type: string, selection?: Parameters<typeof geminiMediaSelectionFields>[0]['selection']) {
    return geminiMediaSelectionFields({ name: 'file', mime_type, selection });
}

describe('geminiMediaSelectionFields', () => {
    it.each([
        ['no selection', 'application/pdf', undefined, {}],
        [
            'a page range',
            'application/pdf',
            { pages: { start: 3, end: 7 } },
            { documentMetadata: { startPage: 3, endPage: 7 } },
        ],
        [
            'an open page range with text layers',
            DOCX,
            { pages: { start: 2 }, layers: ['page_text'] },
            {
                documentMetadata: { startPage: 2, contentLayers: ['PAGE_TEXT'] },
            },
        ],
        [
            'a sheet and cell range',
            XLSX,
            { sheet: { index: 2, cells: 'A1:B2' } },
            {
                spreadsheetMetadata: { sheetIndex: 2, visibleCells: 'A1:B2' },
            },
        ],
        [
            'an audio range',
            'audio/wav',
            { time: { start: 60, end: 150.5 } },
            {
                audioMetadata: { startOffset: '60s', endOffset: '150.5s' },
            },
        ],
        [
            'a video range with layers',
            'video/mp4',
            { time: { end: 30 }, layers: ['frames', 'TRANSCRIPT'] },
            {
                videoMetadata: { endOffset: '30s', contentLayers: ['FRAMES', 'TRANSCRIPT'] },
            },
        ],
        [
            'document layers alone',
            'application/pdf',
            { layers: ['page_text'] },
            {
                documentMetadata: { contentLayers: ['PAGE_TEXT'] },
            },
        ],
        [
            'spreadsheet layers alone',
            XLSX,
            { layers: ['visual'] },
            { spreadsheetMetadata: { contentLayers: ['VISUAL'] } },
        ],
        ['video layers alone', 'video/mp4', { layers: ['frames'] }, { videoMetadata: { contentLayers: ['FRAMES'] } }],
        // The values pass through unchanged so Gemini validates them, including a page range on a non-document.
        ['a page range on an image', 'image/png', { pages: { start: 1 } }, { documentMetadata: { startPage: 1 } }],
    ])('maps %s', (_name, mime, selection, expected) => {
        expect(fields(mime, selection)).toEqual(expected);
    });

    it.each([
        ['two kinds of range', 'application/pdf', { pages: { start: 1 }, time: { start: 1 } }, 'more than one'],
        ['a time range on a document', 'application/pdf', { time: { start: 1 } }, 'applies to audio or video'],
        ['layers on audio', 'audio/wav', { time: { start: 1 }, layers: ['audio'] }, 'no content layers for audio'],
        ['layers alone on audio', 'audio/wav', { layers: ['audio'] }, 'no content layers for audio'],
    ])('rejects %s', (_name, mime, selection, message) => {
        expect(() => fields(mime, selection)).toThrow(message);
    });
});

describe('VertexAIDriver file selections', () => {
    class Driver extends VertexAIDriver {
        selects(model: string): boolean {
            return this.supportsMediaSelection({ model });
        }
    }
    const driver = new Driver({ project: 'test', region: 'global' });

    it.each([
        ['locations/global/publishers/google/models/gemini-4-neon', true],
        ['publishers/google/models/gemini-3.8-flash', true],
        ['locations/global/publishers/anthropic/models/claude-haiku-5-5', false],
        ['publishers/google/models/gemini-omni-1.2-flash', false],
        ['publishers/google/models/imagen-4.0-generate-001', false],
    ])('%s -> %s', (model, expected) => {
        expect(driver.selects(model)).toBe(expected);
    });
});

describe('@google/genai request converters', () => {
    // The SDK copies Part fields one by one into the Vertex request and drops any it does not know, with no error.
    // Releases before 3.5.0 drop these four, so the model would read the whole file. This test sends a request
    // through the installed SDK and fails until it forwards them.
    it('forward the selection metadata of user parts and tool-result parts to Vertex', async () => {
        const fetch = vi.fn(async () => Response.json({ candidates: [{ content: { role: 'model', parts: [] } }] }));
        const client = new GoogleGenAI({ vertexai: true, apiKey: 'test-key', httpOptions: { fetch } });
        const file = { inlineData: { data: 'AQID', mimeType: 'application/pdf' } };
        await client.models.generateContent({
            model: 'gemini-4-neon',
            contents: [
                { role: 'user', parts: [{ ...file, documentMetadata: { startPage: 2, endPage: 2 } }] },
                {
                    role: 'user',
                    parts: [
                        {
                            functionResponse: {
                                name: 'open_report',
                                response: {},
                                parts: [{ ...file, spreadsheetMetadata: { sheetIndex: 2 }, audioMetadata: {} }],
                            },
                        },
                    ],
                },
            ],
        } as Parameters<GoogleGenAI['models']['generateContent']>[0]);

        const [, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
        const body = JSON.parse(String(init.body));
        expect(body.contents[0].parts[0].documentMetadata).toEqual({ startPage: 2, endPage: 2 });
        expect(body.contents[1].parts[0].functionResponse.parts[0].spreadsheetMetadata).toEqual({ sheetIndex: 2 });
    });
});
