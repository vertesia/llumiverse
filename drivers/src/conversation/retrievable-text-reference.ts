import {
    canonicalJsonContentString,
    type ExternalReferenceBlock,
    resolveActiveTextExternalReference,
} from '@llumiverse/conversation';

const MAX_REFERENCE_PREVIEW_LENGTH = 512;
const MAX_RETRIEVAL_ARGUMENTS_LENGTH = 2048;

/** Project only the accepted, active retrieval cue; the archived source bytes remain external. */
export function retrievableTextReference(document: unknown, block: ExternalReferenceBlock): string {
    const resolved = resolveActiveTextExternalReference(document, block.asset_id, block.id);
    const preview = resolved.block.preview;
    if (preview === undefined || preview.length > MAX_REFERENCE_PREVIEW_LENGTH) {
        throw new TypeError(`Canonical external reference ${block.id} lacks a bounded preview`);
    }
    const args = canonicalJsonContentString(resolved.block.retrieval.arguments);
    if (args.length > MAX_RETRIEVAL_ARGUMENTS_LENGTH) {
        throw new TypeError(`Canonical external reference ${block.id} has oversized retrieval arguments`);
    }
    return `Preview: ${preview}\n[Full original text is available through ${resolved.tool_definition.name} with ${args}.]`;
}
