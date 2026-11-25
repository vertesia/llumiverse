import {
    type DataSource,
    formatOmittedAttachmentNote,
    isToolResultAttachmentSupported,
    normalizeAttachmentMimeType,
    type ToolResultAttachmentApi,
} from '@llumiverse/common';

interface WarnLogger {
    warn: (data: Record<string, unknown>, message: string) => void;
}

export interface ToolResultFiles {
    /** The attachments the model can read, in their original order. */
    files: DataSource[];
    /** One note per omitted attachment, to append to the tool result text. */
    notes: string[];
}

/**
 * Splits a tool result's attachments into those the model can read and a note for each of the others
 * (see `isToolResultAttachmentSupported`).
 *
 * Exception to "preserve supplied input": a tool result stays in the conversation history, so an attachment the
 * provider rejects would fail every later turn of the conversation. The omission is logged with the file name and
 * type only.
 */
export function splitToolResultFiles(
    files: DataSource[] | undefined,
    api: ToolResultAttachmentApi,
    model: string,
    logger?: WarnLogger,
): ToolResultFiles {
    const kept: DataSource[] = [];
    const notes: string[] = [];
    const omitted: Array<{ file_name: string; mime_type: string }> = [];
    for (const file of files ?? []) {
        if (isToolResultAttachmentSupported(api, file.mime_type, model)) {
            kept.push(file);
        } else {
            notes.push(formatOmittedAttachmentNote(file, model));
            omitted.push({ file_name: file.name, mime_type: normalizeAttachmentMimeType(file.mime_type) });
        }
    }
    if (omitted.length > 0) {
        logger?.warn(
            { model, api, omitted_attachments: omitted },
            '[llumiverse] Tool result attachments the model cannot read were replaced with a note',
        );
    }
    return { files: kept, notes };
}

/** The tool result text followed by the notes, one per line. */
export function withAttachmentNotes(text: string | undefined, notes: string[]): string {
    if (notes.length === 0) return text ?? '';
    return text ? `${text}\n${notes.join('\n')}` : notes.join('\n');
}
