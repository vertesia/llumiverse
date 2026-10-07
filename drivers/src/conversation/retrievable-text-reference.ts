import {
    canonicalJsonContentString,
    type ExternalReferenceBlock,
    resolveActiveTextExternalReference,
    resolveIndexedTextExternalReference,
} from '@llumiverse/conversation';

const MAX_REFERENCE_PREVIEW_LENGTH = 512;
const MAX_RETRIEVAL_ARGUMENTS_LENGTH = 2048;

/** Project only the accepted, active retrieval cue; the archived source bytes remain external. */
export function retrievableTextReference(document: unknown, block: ExternalReferenceBlock): string {
    const descriptor =
        typeof document === 'object' && document !== null
            ? Object.getOwnPropertyDescriptor(document, 'indexed_reference_evidence')
            : undefined;
    if (descriptor !== undefined && !Object.hasOwn(descriptor, 'value'))
        throw new TypeError('Indexed retrieval evidence must be an owned value');
    const resolved =
        descriptor === undefined
            ? resolveActiveTextExternalReference(document, block.asset_id, block.id)
            : resolveIndexedTextExternalReference(descriptor.value, block.asset_id, block.id);
    const preview = resolved.block.preview;
    if (preview === undefined || preview.length > MAX_REFERENCE_PREVIEW_LENGTH) {
        throw new TypeError(`Canonical external reference ${block.id} lacks a bounded preview`);
    }
    const args = canonicalJsonContentString(resolved.block.retrieval.arguments);
    if (args.length > MAX_RETRIEVAL_ARGUMENTS_LENGTH) {
        throw new TypeError(`Canonical external reference ${block.id} has oversized retrieval arguments`);
    }
    const original = resolved.asset.kind === 'json' ? 'JSON' : 'text';
    return `Preview: ${preview}\n[Full original ${original} is available through ${resolved.tool_definition.name} with ${args}.]`;
}
