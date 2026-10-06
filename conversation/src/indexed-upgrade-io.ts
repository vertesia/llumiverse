import type { IndexedConversationRecordStore } from './indexed-conversation.js';
import {
    INDEXED_PROCESSING_MAX_IO_BYTES,
    INDEXED_PROCESSING_MAX_PAGE_READS,
    INDEXED_PROCESSING_MAX_RECORD_READS,
} from './schemas/indexed-head.js';

/** All nested page/record reads, immutable writes and verification readbacks spend one step budget. */
export const INDEXED_UPGRADE_STEP_LIMITS = Object.freeze({
    page_reads: 256,
    record_reads: 64,
    writes: 256,
    bytes: 16 * 1024 * 1024,
});

/** This separately bounded phase reads only the active window, never lifetime history. */
export const INDEXED_UPGRADE_ACTIVE_LIMITS = Object.freeze({
    page_reads: INDEXED_PROCESSING_MAX_PAGE_READS,
    record_reads: INDEXED_PROCESSING_MAX_RECORD_READS,
    writes: 256,
    bytes: INDEXED_PROCESSING_MAX_IO_BYTES,
});

export class IndexedConversationUpgradeResourceError extends RangeError {
    constructor(message: string) {
        super(message);
        this.name = 'IndexedConversationUpgradeResourceError';
    }
}

export interface IndexedUpgradeStepIo {
    page_reads: number;
    record_reads: number;
    writes: number;
    bytes: number;
}

export function createIndexedUpgradeStepStore(
    source: IndexedConversationRecordStore,
    limits: Readonly<IndexedUpgradeStepIo> = INDEXED_UPGRADE_STEP_LIMITS,
    precedingUsage?: Readonly<IndexedUpgradeStepIo>,
): {
    store: IndexedConversationRecordStore;
    usage: Readonly<IndexedUpgradeStepIo>;
} {
    const usage: IndexedUpgradeStepIo = precedingUsage
        ? { ...precedingUsage }
        : { page_reads: 0, record_reads: 0, writes: 0, bytes: 0 };
    for (const kind of ['page_reads', 'record_reads', 'writes', 'bytes'] as const)
        if (!Number.isSafeInteger(usage[kind]) || usage[kind] < 0 || usage[kind] > limits[kind])
            throw new IndexedConversationUpgradeResourceError('Indexed upgrade preceding IO exceeds phase budget');
    const spend = (kind: 'page_reads' | 'record_reads' | 'writes', bytes: number) => {
        if (!Number.isSafeInteger(bytes) || bytes < 0)
            throw new IndexedConversationUpgradeResourceError('Indexed upgrade IO has invalid declared bytes');
        if (usage[kind] >= limits[kind] || bytes > limits.bytes - usage.bytes)
            throw new IndexedConversationUpgradeResourceError('Indexed upgrade step exceeds its complete IO budget');
        usage[kind] += 1;
        usage.bytes += bytes;
    };
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            spend('page_reads', ref.size_bytes);
            const bytes = await source.read(ref);
            if (bytes.byteLength !== ref.size_bytes)
                throw new IndexedConversationUpgradeResourceError(
                    'Indexed upgrade page differs from declared byte bound',
                );
            return bytes;
        },
        async write(bytes, ref) {
            if (bytes.byteLength !== ref.size_bytes)
                throw new IndexedConversationUpgradeResourceError(
                    'Indexed upgrade page write differs from declared bytes',
                );
            spend('writes', bytes.byteLength);
            await source.write(bytes, ref);
        },
        async readRecord(value) {
            spend('record_reads', value.size_bytes);
            const bytes = await source.readRecord(value);
            if (bytes.byteLength !== value.size_bytes)
                throw new IndexedConversationUpgradeResourceError(
                    'Indexed upgrade record differs from declared byte bound',
                );
            return bytes;
        },
        async writeRecord(value, bytes) {
            if (bytes.byteLength !== value.size_bytes)
                throw new IndexedConversationUpgradeResourceError(
                    'Indexed upgrade record write differs from declared bytes',
                );
            spend('writes', bytes.byteLength);
            await source.writeRecord(value, bytes);
        },
    };
    return { store, usage };
}
