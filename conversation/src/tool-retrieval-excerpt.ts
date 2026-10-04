import { z } from 'zod';
import { hashContentBytes } from './content-integrity.js';
import { resolveActiveTextExternalReference } from './external-reference-retrieval.js';
import { fingerprintJson } from './identity.js';
import { ToolResultBlockSchema } from './schemas/content.js';
import { ExecutionReceiptSchema } from './schemas/execution.js';
import { assertToolResultReceiptFingerprint } from './tool-result-integrity.js';
import type {
    ConversationDocument,
    ExecutionReceipt,
    ToolCallSourceRef,
    ToolResultBlock,
    ToolRetrievalExcerpt,
} from './types.js';
import { parseConversationDocument } from './validation.js';

/** Check retained cross-record facts. Hosts must separately prove actual byte custody before minting metadata. */
export async function assertToolRetrievalExcerptBinding(
    sourceInput: ConversationDocument,
    resultInput: ToolResultBlock,
    receiptInput: ExecutionReceipt,
): Promise<void> {
    const source = parseConversationDocument(sourceInput);
    const result = ToolResultBlockSchema.parse(resultInput);
    const receipt = ExecutionReceiptSchema.parse(receiptInput);
    const excerpt = receipt.metadata?.retrieval_excerpt;
    if (
        !excerpt ||
        receipt.executor !== 'application' ||
        receipt.status !== 'success' ||
        result.status !== 'success' ||
        receipt.call_id !== result.call_id ||
        !receipt.call_source ||
        receipt.call_source.conversation.conversation_id !== source.id ||
        receipt.call_source.conversation.revision > source.revision ||
        excerpt.source.conversation_id !== source.id ||
        excerpt.source.revision !== receipt.call_source.conversation.revision
    )
        throw new Error('Retrieval excerpt requires an exact successful application read source');
    const turn = source.turns.find((value) => value.id === receipt.call_source?.turn_id);
    const call = turn?.blocks.find((value) => value.id === receipt.call_source?.block_id);
    const reference = resolveActiveTextExternalReference(source, excerpt.asset_id, excerpt.external_reference_block_id);
    const returned = result.content.find((value) => value.id === excerpt.returned_block_id);
    if (
        call?.type !== 'tool_call' ||
        call.executor !== 'application' ||
        call.call_id !== receipt.call_id ||
        call.tool_name !== reference.tool_definition.name ||
        call.definition_id !== reference.tool_definition.id ||
        (await fingerprintJson(call)) !== receipt.call_source.call_fingerprint ||
        reference.tool_definition.id !== excerpt.tool_definition_id ||
        reference.accepted_asset_operation_id !== excerpt.accepted_asset_operation_id ||
        reference.asset.content_hash !== excerpt.content_hash ||
        (await fingerprintJson(reference.block)) !== excerpt.reference_fingerprint ||
        reference.asset.byte_length === undefined ||
        excerpt.byte_start > excerpt.byte_end_exclusive ||
        excerpt.byte_end_exclusive > reference.asset.byte_length ||
        returned?.type !== 'text' ||
        (await fingerprintJson(returned)) !== excerpt.returned_block_fingerprint
    )
        throw new Error('Retrieval excerpt changed its exact read definition, range or returned block');
    if (excerpt.projection.kind === 'json_byte_excerpt') {
        const content: unknown = JSON.parse(returned.text);
        if (
            content === null ||
            typeof content !== 'object' ||
            Array.isArray(content) ||
            Reflect.get(content, 'asset_id') !== excerpt.asset_id ||
            Reflect.get(content, 'content_hash') !== excerpt.content_hash ||
            Reflect.get(content, 'byte_start') !== excerpt.byte_start ||
            Reflect.get(content, 'byte_end_exclusive') !== excerpt.byte_end_exclusive ||
            Reflect.get(content, 'total_bytes') !== reference.asset.byte_length ||
            typeof Reflect.get(content, 'content') !== 'string'
        )
            throw new Error('Retrieval byte excerpt changed its returned range');
        const text = Reflect.get(content, 'content');
        if (
            typeof text !== 'string' ||
            new TextEncoder().encode(text).byteLength !== excerpt.byte_end_exclusive - excerpt.byte_start
        )
            throw new Error('Retrieval byte excerpt does not contain its exact returned UTF-8 byte length');
    }
    await assertToolResultReceiptFingerprint(result, receipt);
}

const ReadArguments = z.strictObject({
    path: z.string().min(1),
    asset_id: z.string().min(1),
    start_byte: z.number().int().nonnegative().safe().optional(),
    byte_count: z.number().int().min(1).max(5000).optional(),
    start_line: z.number().int().min(1).safe().optional(),
    end_line: z.number().int().min(1).safe().optional(),
    line_numbers: z.boolean().default(false),
});

export async function renderCanonicalToolRetrievalResult(
    document: ConversationDocument,
    source: ToolCallSourceRef,
    assetId: string,
    bytes: Uint8Array,
    hydratedArguments?: unknown,
): Promise<{
    reference: ReturnType<typeof resolveActiveTextExternalReference>;
    byte_start: number;
    byte_end_exclusive: number;
    projection: ToolRetrievalExcerpt['projection'];
    text: string;
}> {
    const turn = document.turns.find((item) => item.id === source.turn_id);
    const call = turn?.blocks.find((item) => item.id === source.block_id);
    if (
        source.conversation.conversation_id !== document.id ||
        source.conversation.revision > document.revision ||
        call?.type !== 'tool_call' ||
        call.executor !== 'application' ||
        call.call_id !== source.call_id ||
        (await fingerprintJson(call)) !== source.call_fingerprint
    )
        throw new TypeError('Canonical retrieval requires its exact immutable application call');
    if (
        call.arguments.type !== 'json' &&
        (call.arguments.type !== 'externalized_json' || hydratedArguments === undefined)
    )
        throw new TypeError('Canonical retrieval requires independently hydrated execution arguments');
    const execution = { call, arguments: call.arguments.type === 'json' ? call.arguments.value : hydratedArguments };
    const args = ReadArguments.parse(execution.arguments);
    const reference = resolveActiveTextExternalReference(document, assetId);
    const integrity = await hashContentBytes(bytes);
    const locator = reference.asset.storage.type === 'external' ? reference.asset.storage.locator : undefined;
    if (
        execution.call.tool_name !== reference.tool_definition.name ||
        execution.call.definition_id !== reference.tool_definition.id ||
        args.asset_id !== assetId ||
        reference.block.retrieval.capability !== 'read_artifact' ||
        reference.block.retrieval.version !== 1 ||
        reference.block.retrieval.arguments.asset_id !== assetId ||
        reference.block.retrieval.arguments.path !== args.path ||
        reference.asset.storage.type !== 'external' ||
        reference.asset.storage.resolver !== 'vertesia.agent_artifact' ||
        locator?.artifact_path !== args.path ||
        reference.asset.byte_length !== bytes.byteLength ||
        !reference.asset.content_hash ||
        reference.asset.content_hash !== integrity.content_hash
    )
        throw new TypeError('Canonical retrieval is not the exact accepted asset read capability');
    if (
        (args.start_byte !== undefined || args.byte_count !== undefined) &&
        (args.start_line !== undefined || args.end_line !== undefined || args.line_numbers)
    )
        throw new TypeError('Canonical retrieval cannot mix byte and line projections');
    if (args.end_line !== undefined && args.end_line < (args.start_line ?? 1))
        throw new RangeError('Canonical retrieval line range is invalid');
    if (
        args.start_byte !== undefined ||
        args.byte_count !== undefined ||
        (args.start_line === undefined && args.end_line === undefined && !args.line_numbers)
    ) {
        const start = args.start_byte ?? 0;
        if (start > bytes.length) throw new RangeError('Canonical retrieval byte range is invalid');
        let end = Math.min(bytes.length, start + (args.byte_count ?? 5000));
        let content = '';
        while (end > start) {
            try {
                content = new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(start, end));
                break;
            } catch {
                end -= 1;
            }
        }
        if (end === start && start < bytes.length) throw new RangeError('Canonical retrieval is not a UTF-8 boundary');
        return {
            reference,
            byte_start: start,
            byte_end_exclusive: end,
            projection: { kind: 'json_byte_excerpt' as const },
            text: JSON.stringify({
                asset_id: assetId,
                content_hash: reference.asset.content_hash,
                content,
                byte_start: start,
                byte_end_exclusive: end,
                total_bytes: bytes.length,
                ...(end < bytes.length ? { next_byte_offset: end } : {}),
            }),
        };
    }
    const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    const delimiter = text.includes('\r\n') ? '\r\n' : '\n';
    const lines = text.split(delimiter);
    const startIdx = Math.max(0, (args.start_line ?? 1) - 1);
    let endIdx =
        args.end_line === undefined ? Math.min(lines.length, startIdx + 2000) : Math.min(lines.length, args.end_line);
    endIdx = Math.max(startIdx, endIdx);
    const encoder = new TextEncoder();
    const selected: string[] = [];
    let selectedBytes = 0;
    let byteCapped = false;
    let withinLineByteEnd: number | undefined;
    for (let index = startIdx; index < endIdx; index++) {
        const encoded = encoder.encode(lines[index]);
        const nextBytes = selectedBytes + (selected.length > 0 ? 1 : 0) + encoded.byteLength;
        if (nextBytes <= 5000) {
            selected.push(lines[index]);
            selectedBytes = nextBytes;
            continue;
        }
        byteCapped = true;
        if (selected.length > 0) break;
        let byteEnd = 5000;
        while (byteEnd > 0 && (encoded[byteEnd] & 0xc0) === 0x80) byteEnd--;
        selected.push(new TextDecoder('utf-8', { fatal: true }).decode(encoded.subarray(0, byteEnd)));
        withinLineByteEnd = byteEnd;
        break;
    }
    endIdx = startIdx + selected.length;
    const truncated = endIdx < lines.length || byteCapped;
    let continuationByteOffset: number | undefined;
    if (withinLineByteEnd !== undefined) {
        continuationByteOffset = withinLineByteEnd;
        const delimiterBytes = encoder.encode(delimiter).byteLength;
        for (let index = 0; index < startIdx; index++) {
            continuationByteOffset += encoder.encode(lines[index]).byteLength + delimiterBytes;
        }
    }
    const slice = {
        content: args.line_numbers
            ? selected.map((line, index) => `${String(startIdx + index + 1).padStart(4, ' ')}\t${line}`).join('\n')
            : selected.join('\n'),
        start_line: startIdx + 1,
        end_line: endIdx,
        total_lines: lines.length,
        truncated,
        truncation_hint: truncated
            ? continuationByteOffset === undefined
                ? `[Showing lines ${startIdx + 1}-${endIdx} of ${lines.length}. Use start_line=${endIdx + 1} to continue.]`
                : `[Showing lines ${startIdx + 1}-${endIdx} of ${lines.length}. Use start_byte=${continuationByteOffset} to continue this line.]`
            : undefined,
    };
    const content = [
        '---',
        `asset_id: "${assetId}"`,
        `path: "${args.path}"`,
        `content_hash: "${reference.asset.content_hash}"`,
        `lines: ${slice.start_line}-${slice.end_line}`,
        `total_lines: ${slice.total_lines}`,
        `truncated: ${slice.truncated}`,
        '---',
        '',
        slice.content,
        ...(slice.truncation_hint ? ['', slice.truncation_hint] : []),
    ].join('\n');
    return {
        reference,
        // Line formatting consumes the complete authenticated text; it is not a raw byte slice.
        byte_start: 0,
        byte_end_exclusive: bytes.length,
        projection: {
            kind: 'rendered_line_excerpt' as const,
            start_line: slice.start_line,
            end_line: slice.end_line,
            line_numbers: args.line_numbers,
        },
        text: JSON.stringify({ content, asset_id: assetId, content_hash: reference.asset.content_hash }),
    };
}

/** Byte custody is established only by the host's authenticated immutable asset reader. */
export async function verifyToolRetrievalExcerptBytes(
    document: ConversationDocument,
    result: ToolResultBlock,
    receipt: ExecutionReceipt,
    bytes: Uint8Array,
    hydratedArguments?: unknown,
): Promise<void> {
    await assertToolRetrievalExcerptBinding(document, result, receipt);
    const claim = receipt.metadata?.retrieval_excerpt;
    if (!claim || !receipt.call_source) throw new TypeError('Retrieval receipt is unavailable');
    const expected = await renderCanonicalToolRetrievalResult(
        document,
        receipt.call_source,
        claim.asset_id,
        bytes,
        hydratedArguments,
    );
    const block = result.content.find((value) => value.id === claim.returned_block_id);
    if (
        block?.type !== 'text' ||
        block.text !== expected.text ||
        claim.byte_start !== expected.byte_start ||
        claim.byte_end_exclusive !== expected.byte_end_exclusive ||
        (await fingerprintJson(claim.projection)) !== (await fingerprintJson(expected.projection))
    )
        throw new TypeError('Retrieval receipt differs from authenticated asset range bytes');
}
