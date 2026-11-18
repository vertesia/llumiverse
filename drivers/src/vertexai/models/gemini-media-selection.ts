import type {
    AudioMetadata,
    DocumentMetadata,
    DocumentMetadataContentLayer,
    SpreadsheetMetadata,
    SpreadsheetMetadataContentLayer,
    VideoMetadata,
    VideoMetadataContentLayer,
} from '@google/genai';
import type { DataSource } from '@llumiverse/core';

/**
 * The Gemini `Part` and `FunctionResponsePart` fields that select part of a file. @google/genai 3.5.0 is the first
 * release whose Vertex request converters forward them; earlier releases drop them without an error.
 * https://ai.google.dev/gemini-api/docs/document-processing#customize-document-processing
 * https://ai.google.dev/gemini-api/docs/audio#clipping-intervals
 */
export interface GeminiMediaSelectionFields {
    documentMetadata?: DocumentMetadata;
    spreadsheetMetadata?: SpreadsheetMetadata;
    audioMetadata?: AudioMetadata;
    videoMetadata?: VideoMetadata;
}

const SPREADSHEET_MIME_TYPES = new Set([
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel',
    'application/vnd.oasis.opendocument.spreadsheet',
    'text/csv',
]);

function offset(seconds: number | undefined): string | undefined {
    return seconds === undefined ? undefined : `${seconds}s`;
}

/**
 * Gemini spells content layers in upper case (`PAGE_TEXT`); llumiverse accepts either case. Names outside the SDK
 * enums are passed through, typed as that enum, so Gemini validates them and newer layers work without a release.
 */
function contentLayers<Layer extends string>(layers: string[] | undefined): Layer[] | undefined {
    return layers?.map((layer) => layer.toUpperCase() as Layer);
}

function withoutUndefined<T extends object>(value: T): T {
    return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== undefined)) as T;
}

/**
 * Translates a file's `selection` into the Gemini part metadata for its kind: pages to `documentMetadata`, a sheet
 * to `spreadsheetMetadata`, and a time range to `audioMetadata` or `videoMetadata` by MIME type. Layers alone follow
 * the file type. The values are passed as given so Gemini validates them; only a selection that has no Gemini field
 * (two kinds of range at once, a time range on a document, layers on audio) is rejected here.
 */
export function geminiMediaSelectionFields(
    file: Pick<DataSource, 'name' | 'mime_type' | 'selection'>,
): GeminiMediaSelectionFields {
    const selection = file.selection;
    if (!selection) return {};
    const { pages, sheet, time, layers } = selection;
    const ranges = [pages, sheet, time].filter((range) => range !== undefined).length;
    if (ranges > 1) {
        throw new Error(`The selection of "${file.name}" sets more than one of pages, sheet and time; set one`);
    }
    const mime = file.mime_type.toLowerCase();
    const isAudio = mime.startsWith('audio/');
    const isVideo = mime.startsWith('video/');

    if (pages) {
        return {
            documentMetadata: withoutUndefined({
                startPage: pages.start,
                endPage: pages.end,
                contentLayers: contentLayers<DocumentMetadataContentLayer>(layers),
            }),
        };
    }
    if (sheet) {
        return {
            spreadsheetMetadata: withoutUndefined({
                sheetIndex: sheet.index,
                visibleCells: sheet.cells,
                contentLayers: contentLayers<SpreadsheetMetadataContentLayer>(layers),
            }),
        };
    }
    if (time) {
        const range = { startOffset: offset(time.start), endOffset: offset(time.end) };
        if (isVideo) {
            return {
                videoMetadata: withoutUndefined({
                    ...range,
                    contentLayers: contentLayers<VideoMetadataContentLayer>(layers),
                }),
            };
        }
        if (isAudio) {
            if (layers) throw new Error(`Gemini has no content layers for audio; remove layers from "${file.name}"`);
            return { audioMetadata: withoutUndefined(range) };
        }
        throw new Error(`A time range applies to audio or video, but "${file.name}" is ${file.mime_type}`);
    }
    if (!layers) return {};
    if (isVideo) return { videoMetadata: { contentLayers: contentLayers<VideoMetadataContentLayer>(layers) } };
    if (isAudio) throw new Error(`Gemini has no content layers for audio; remove layers from "${file.name}"`);
    if (SPREADSHEET_MIME_TYPES.has(mime)) {
        return { spreadsheetMetadata: { contentLayers: contentLayers<SpreadsheetMetadataContentLayer>(layers) } };
    }
    return { documentMetadata: { contentLayers: contentLayers<DocumentMetadataContentLayer>(layers) } };
}
