import type { z } from 'zod';
import { ConversationValidationError } from './diagnostics.js';
import { type JsonInputLimits, preflightJsonInput } from './json-preflight.js';
import { CONVERSATION_EXPERIMENTAL_REVISION, CONVERSATION_SCHEMA_VERSION } from './runtime-constants.js';
import { ConversationTurnSchema } from './schemas/content.js';
import {
    CONVERSATION_TRANSCRIPT_FORMAT,
    CONVERSATION_TRANSCRIPT_MAX_ASSETS,
    CONVERSATION_TRANSCRIPT_MAX_BLOCKS_PER_TURN,
    CONVERSATION_TRANSCRIPT_MAX_GENERATIONS,
    CONVERSATION_TRANSCRIPT_MAX_OMISSIONS,
    type ConversationTranscriptAgentBlockSchema,
    type ConversationTranscriptAssetOmissionSchema,
    type ConversationTranscriptAssetSchema,
    type ConversationTranscriptBlockOmissionSchema,
    type ConversationTranscriptExternalReferenceBlockSchema,
    ConversationTranscriptFragmentSchema,
    type ConversationTranscriptGenerationInputOmissionSchema,
    type ConversationTranscriptGenerationOmissionSchema,
    type ConversationTranscriptGenerationSchema,
    type ConversationTranscriptInputOmissionSchema,
    type ConversationTranscriptProgramBlockSchema,
    ConversationTranscriptProjectionInputSchema,
    type ConversationTranscriptRenderableBlockSchema,
    type ConversationTranscriptToolArgumentsSchema,
    type ConversationTranscriptToolResultBlockSchema,
    type ConversationTranscriptTurnOmissionSchema,
    type ConversationTranscriptTurnSchema,
    type ConversationTranscriptUserBlockSchema,
} from './schemas/transcript.js';
import type {
    AgentContentBlock,
    Asset,
    ConversationTurn,
    Generation,
    NestedToolResultContentBlock,
    ProgramContentBlock,
    ToolArguments,
    UserContentBlock,
} from './types.js';
import { diagnosticsFromZodError } from './validation.js';

export type ConversationTranscriptProjectionInput = z.infer<typeof ConversationTranscriptProjectionInputSchema>;
export type ConversationTranscriptExternalReferenceBlock = z.infer<
    typeof ConversationTranscriptExternalReferenceBlockSchema
>;
export type ConversationTranscriptFragment = z.infer<typeof ConversationTranscriptFragmentSchema>;
export type ConversationTranscriptTurn = z.infer<typeof ConversationTranscriptTurnSchema>;
export type ConversationTranscriptAsset = z.infer<typeof ConversationTranscriptAssetSchema>;
export type ConversationTranscriptGeneration = z.infer<typeof ConversationTranscriptGenerationSchema>;
export type ConversationTranscriptUserBlock = z.infer<typeof ConversationTranscriptUserBlockSchema>;
export type ConversationTranscriptAgentBlock = z.infer<typeof ConversationTranscriptAgentBlockSchema>;
export type ConversationTranscriptProgramBlock = z.infer<typeof ConversationTranscriptProgramBlockSchema>;
export type ConversationTranscriptRenderableBlock = z.infer<typeof ConversationTranscriptRenderableBlockSchema>;
export type ConversationTranscriptToolResultBlock = z.infer<typeof ConversationTranscriptToolResultBlockSchema>;
export type ConversationTranscriptToolArguments = z.infer<typeof ConversationTranscriptToolArgumentsSchema>;
export type ConversationTranscriptInputOmission = z.infer<typeof ConversationTranscriptInputOmissionSchema>;
export type ConversationTranscriptGenerationInputOmission = z.infer<
    typeof ConversationTranscriptGenerationInputOmissionSchema
>;
export type ConversationTranscriptTurnOmission = z.infer<typeof ConversationTranscriptTurnOmissionSchema>;
export type ConversationTranscriptBlockOmission = z.infer<typeof ConversationTranscriptBlockOmissionSchema>;
export type ConversationTranscriptAssetOmission = z.infer<typeof ConversationTranscriptAssetOmissionSchema>;
export type ConversationTranscriptGenerationOmission = z.infer<typeof ConversationTranscriptGenerationOmissionSchema>;

export interface ConversationTranscriptProjectionOptions {
    json_input_limits?: Partial<JsonInputLimits>;
    /** Host-owned allowlist only. Values are copied exclusively from the exact selected source turn. */
    select_turn_metadata_keys?: (turn: Readonly<Pick<ConversationTurn, 'id' | 'kind'>>) => readonly string[];
}

export type ConversationTranscriptProjectionErrorCode =
    | 'duplicate_reference'
    | 'invalid_fragment'
    | 'invalid_source_turn'
    | 'projection_limit_exceeded';

export class ConversationTranscriptProjectionError extends Error {
    constructor(
        readonly code: ConversationTranscriptProjectionErrorCode,
        message: string,
    ) {
        super(message);
        this.name = 'ConversationTranscriptProjectionError';
    }
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function validateShape<T>(schema: z.ZodType<T>, value: unknown, message: string): T {
    const result = schema.safeParse(value);
    if (!result.success) {
        throw new ConversationValidationError(message, diagnosticsFromZodError(result.error));
    }
    return result.data;
}

function assertUnique(values: readonly string[], label: string): void {
    if (new Set(values).size !== values.length) {
        throw new ConversationTranscriptProjectionError(
            'duplicate_reference',
            `${label} contains duplicate identities`,
        );
    }
}

function appendBounded<T>(target: T[], value: T, label: string): void {
    if (target.length >= CONVERSATION_TRANSCRIPT_MAX_OMISSIONS) {
        throw new ConversationTranscriptProjectionError(
            'projection_limit_exceeded',
            `${label} exceeds the bounded transcript omission limit`,
        );
    }
    target.push(value);
}

function rememberOmittedAsset(
    omittedAssets: ConversationTranscriptAssetOmission[],
    includedAssetIds: ReadonlySet<string>,
    omission: ConversationTranscriptAssetOmission,
): void {
    if (includedAssetIds.has(omission.asset_id) || omittedAssets.some((item) => item.asset_id === omission.asset_id)) {
        return;
    }
    appendBounded(omittedAssets, omission, 'omitted assets');
}

function rememberIncludedAsset(
    includedAssetIds: Set<string>,
    omittedAssets: ConversationTranscriptAssetOmission[],
    assetId: string,
): void {
    includedAssetIds.add(assetId);
    const omissionIndex = omittedAssets.findIndex((item) => item.asset_id === assetId);
    if (omissionIndex >= 0) omittedAssets.splice(omissionIndex, 1);
}

function missingModelArgumentsBlock(value: unknown): { turn_id: string; block_id: string } | undefined {
    if (!isRecord(value) || value.kind !== 'agent' || typeof value.id !== 'string' || !Array.isArray(value.blocks)) {
        return undefined;
    }
    for (const block of value.blocks) {
        if (
            isRecord(block) &&
            typeof block.id === 'string' &&
            block.type === 'tool_call' &&
            isRecord(block.arguments) &&
            block.arguments.type === 'externalized_json' &&
            !Object.hasOwn(block.arguments, 'model_value')
        ) {
            return { turn_id: value.id, block_id: block.id };
        }
    }
    return undefined;
}

function sanitizeMissingModelArguments(value: unknown, omittedBlocks: ConversationTranscriptBlockOmission[]): unknown {
    if (!isRecord(value) || value.kind !== 'agent' || !Array.isArray(value.blocks)) return value;
    const missing = value.blocks.filter(
        (block) =>
            isRecord(block) &&
            block.type === 'tool_call' &&
            isRecord(block.arguments) &&
            block.arguments.type === 'externalized_json' &&
            !Object.hasOwn(block.arguments, 'model_value'),
    );
    if (missing.length === 0) return value;
    if (typeof value.id !== 'string' || missing.some((block) => typeof block.id !== 'string')) return value;
    for (const block of missing) {
        appendBounded(
            omittedBlocks,
            {
                turn_id: value.id,
                block_id: block.id as string,
                reason: 'model_visible_arguments_unavailable',
            },
            'omitted blocks',
        );
    }
    const missingIds = new Set(missing.map((block) => block.id));
    return { ...value, blocks: value.blocks.filter((block) => !isRecord(block) || !missingIds.has(block.id)) };
}

function parseSourceTurn(value: unknown, omittedBlocks: ConversationTranscriptBlockOmission[]): ConversationTurn {
    const sanitized = sanitizeMissingModelArguments(value, omittedBlocks);
    const result = ConversationTurnSchema.safeParse(sanitized);
    if (!result.success) {
        const missing = missingModelArgumentsBlock(value);
        throw new ConversationTranscriptProjectionError(
            'invalid_source_turn',
            missing
                ? `Canonical turn ${missing.turn_id} has unavailable model-visible tool arguments`
                : 'Canonical transcript source turn is invalid',
        );
    }
    return result.data;
}

function mediaAssetId(block: { type: string } & Record<string, unknown>): string | undefined {
    switch (block.type) {
        case 'image':
        case 'document':
        case 'audio':
        case 'video':
            return typeof block.asset_id === 'string' ? block.asset_id : undefined;
        default:
            return undefined;
    }
}

function projectMediaBlock(
    turnId: string,
    block: Extract<
        UserContentBlock | AgentContentBlock | ProgramContentBlock | NestedToolResultContentBlock,
        { type: 'image' | 'document' | 'audio' | 'video' }
    >,
    assetsById: ReadonlyMap<string, Asset>,
    includedAssetIds: Set<string>,
    omittedBlocks: ConversationTranscriptBlockOmission[],
    omittedAssets: ConversationTranscriptAssetOmission[],
): typeof block | undefined {
    const asset = assetsById.get(block.asset_id);
    if (asset === undefined) {
        appendBounded(
            omittedBlocks,
            { turn_id: turnId, block_id: block.id, reason: 'referenced_asset_unavailable' },
            'omitted blocks',
        );
        rememberOmittedAsset(omittedAssets, includedAssetIds, {
            asset_id: block.asset_id,
            reason: 'not_supplied',
        });
        return undefined;
    }
    if (asset.kind !== block.type) {
        appendBounded(
            omittedBlocks,
            { turn_id: turnId, block_id: block.id, reason: 'referenced_asset_kind_mismatch' },
            'omitted blocks',
        );
        rememberOmittedAsset(omittedAssets, includedAssetIds, {
            asset_id: block.asset_id,
            reason: 'kind_mismatch',
        });
        return undefined;
    }
    rememberIncludedAsset(includedAssetIds, omittedAssets, block.asset_id);
    return {
        id: block.id,
        type: block.type,
        asset_id: block.asset_id,
        ...(block.caption === undefined ? {} : { caption: block.caption }),
        ...(block.selection === undefined ? {} : { selection: structuredClone(block.selection) }),
    } as typeof block;
}

function projectRenderableBlock(
    turnId: string,
    block: UserContentBlock | AgentContentBlock | ProgramContentBlock | NestedToolResultContentBlock,
    assetsById: ReadonlyMap<string, Asset>,
    includedAssetIds: Set<string>,
    omittedBlocks: ConversationTranscriptBlockOmission[],
    omittedAssets: ConversationTranscriptAssetOmission[],
): ConversationTranscriptRenderableBlock | undefined {
    switch (block.type) {
        case 'text':
            return {
                id: block.id,
                type: block.type,
                text: block.text,
                format: block.format,
                ...(block.language === undefined ? {} : { language: block.language }),
            };
        case 'json':
            return { id: block.id, type: block.type, value: structuredClone(block.value) };
        case 'image':
        case 'document':
        case 'audio':
        case 'video':
            return projectMediaBlock(turnId, block, assetsById, includedAssetIds, omittedBlocks, omittedAssets);
        case 'reasoning':
            return {
                id: block.id,
                type: block.type,
                text: block.text,
                representation: block.representation,
            };
        case 'native_replay':
            appendBounded(
                omittedBlocks,
                { turn_id: turnId, block_id: block.id, reason: 'native_replay' },
                'omitted blocks',
            );
            return undefined;
        case 'extension':
            appendBounded(
                omittedBlocks,
                { turn_id: turnId, block_id: block.id, reason: 'unsupported_extension' },
                'omitted blocks',
            );
            return undefined;
        case 'external_reference': {
            const asset = assetsById.get(block.asset_id);
            if (
                asset?.kind === block.original_type &&
                asset.content_hash !== undefined &&
                block.content_hash === asset.content_hash
            ) {
                const boundedCue = (value: string) => value.slice(0, 512).replace(/[\uD800-\uDBFF]$/u, '');
                return {
                    id: block.id,
                    type: block.type,
                    asset_id: asset.id,
                    original_type: block.original_type,
                    content_hash: asset.content_hash,
                    description: boundedCue(block.description),
                    ...(block.preview === undefined ? {} : { preview: boundedCue(block.preview) }),
                };
            }
            appendBounded(
                omittedBlocks,
                { turn_id: turnId, block_id: block.id, reason: 'unsupported_external_reference' },
                'omitted blocks',
            );
            return undefined;
        }
        case 'tool_call':
            return undefined;
    }
}

function projectToolArguments(argumentsValue: ToolArguments): ConversationTranscriptToolArguments | undefined {
    switch (argumentsValue.type) {
        case 'json':
            return { type: 'json', value: structuredClone(argumentsValue.value) };
        case 'invalid':
            return { type: 'invalid', raw: argumentsValue.raw };
        case 'externalized_json':
            if (!Object.hasOwn(argumentsValue, 'model_value') || argumentsValue.model_value === undefined) {
                return undefined;
            }
            return { type: 'json', value: structuredClone(argumentsValue.model_value) };
    }
}

function projectToolResult(
    turnId: string,
    block: Extract<ConversationTurn, { kind: 'tool' }>['blocks'][number],
    assetsById: ReadonlyMap<string, Asset>,
    includedAssetIds: Set<string>,
    omittedBlocks: ConversationTranscriptBlockOmission[],
    omittedAssets: ConversationTranscriptAssetOmission[],
): ConversationTranscriptToolResultBlock {
    const content = block.content.flatMap((item) => {
        const projected = projectRenderableBlock(
            turnId,
            item,
            assetsById,
            includedAssetIds,
            omittedBlocks,
            omittedAssets,
        );
        return projected === undefined ? [] : [projected];
    });
    if (content.length > CONVERSATION_TRANSCRIPT_MAX_BLOCKS_PER_TURN) {
        throw new ConversationTranscriptProjectionError(
            'projection_limit_exceeded',
            'Projected tool result exceeds the bounded block limit',
        );
    }
    return { id: block.id, type: 'tool_result', call_id: block.call_id, status: block.status, content };
}

function projectTurn(
    turn: ConversationTurn,
    assetsById: ReadonlyMap<string, Asset>,
    includedAssetIds: Set<string>,
    omittedBlocks: ConversationTranscriptBlockOmission[],
    omittedAssets: ConversationTranscriptAssetOmission[],
): ConversationTranscriptTurn {
    const common = {
        id: turn.id,
        kind: turn.kind,
        status: turn.status,
        timestamps: structuredClone(turn.timestamps),
    } as const;
    if (turn.kind === 'tool') {
        return {
            ...common,
            kind: 'tool',
            blocks: [
                projectToolResult(turn.id, turn.blocks[0], assetsById, includedAssetIds, omittedBlocks, omittedAssets),
            ],
        };
    }
    const blocks: ConversationTranscriptAgentBlock[] = [];
    for (const block of turn.blocks) {
        if ((turn.kind === 'agent' || turn.kind === 'program') && block.type === 'tool_call') {
            const projectedArguments = projectToolArguments(block.arguments);
            if (projectedArguments === undefined) {
                appendBounded(
                    omittedBlocks,
                    { turn_id: turn.id, block_id: block.id, reason: 'model_visible_arguments_unavailable' },
                    'omitted blocks',
                );
                continue;
            }
            blocks.push({
                id: block.id,
                type: 'tool_call',
                call_id: block.call_id,
                tool_name: block.tool_name,
                executor: block.executor,
                arguments: projectedArguments,
            });
            continue;
        }
        const projected = projectRenderableBlock(
            turn.id,
            block,
            assetsById,
            includedAssetIds,
            omittedBlocks,
            omittedAssets,
        );
        if (projected !== undefined) blocks.push(projected);
    }
    if (blocks.length > CONVERSATION_TRANSCRIPT_MAX_BLOCKS_PER_TURN) {
        throw new ConversationTranscriptProjectionError(
            'projection_limit_exceeded',
            `Projected turn ${turn.id} exceeds the bounded block limit`,
        );
    }
    switch (turn.kind) {
        case 'user':
            return { ...common, kind: 'user', blocks: blocks as ConversationTranscriptUserBlock[] };
        case 'agent':
            return {
                ...common,
                kind: 'agent',
                ...('generation_id' in turn && turn.generation_id !== undefined
                    ? { generation_id: turn.generation_id }
                    : {}),
                blocks: blocks as ConversationTranscriptAgentBlock[],
            };
        case 'program':
            return { ...common, kind: 'program', blocks };
    }
}

function projectGeneration(generation: Generation): ConversationTranscriptGeneration {
    const usage = generation.usage;
    const normalizedUsage =
        usage === undefined ? undefined : (({ reported_usage: _reportedUsage, ...safeUsage }) => safeUsage)(usage);
    const optional = {
        ...(generation.resolved_model === undefined ? {} : { resolved_model: generation.resolved_model }),
        ...(generation.finish_reason === undefined ? {} : { finish_reason: generation.finish_reason }),
        ...(normalizedUsage === undefined ? {} : { usage: structuredClone(normalizedUsage) }),
    };
    if (generation.record_source === 'executed') {
        return {
            id: generation.id,
            record_source: generation.record_source,
            purpose: generation.purpose,
            requested_model: generation.requested_model,
            ...optional,
            provider: generation.provider,
            protocol: generation.protocol,
            status: generation.status,
            timestamps: structuredClone(generation.timestamps),
        };
    }
    return {
        id: generation.id,
        record_source: generation.record_source,
        ...(generation.purpose === undefined ? {} : { purpose: generation.purpose }),
        ...(generation.requested_model === undefined ? {} : { requested_model: generation.requested_model }),
        ...optional,
        ...(generation.provider === undefined ? {} : { provider: generation.provider }),
        ...(generation.protocol === undefined ? {} : { protocol: generation.protocol }),
        status: generation.status,
        timestamps: structuredClone(generation.timestamps),
    };
}

function projectAsset(asset: Asset): ConversationTranscriptAsset {
    return {
        id: asset.id,
        kind: asset.kind,
        mime_type: asset.mime_type,
        storage: structuredClone(asset.storage),
        ...(asset.byte_length === undefined ? {} : { byte_length: asset.byte_length }),
        ...(asset.content_hash === undefined ? {} : { content_hash: asset.content_hash }),
        ...(asset.media === undefined ? {} : { media: structuredClone(asset.media) }),
        created_at: asset.created_at,
    };
}

function parseProjectionInput(
    input: unknown,
    omittedBlocks: ConversationTranscriptBlockOmission[],
    options: ConversationTranscriptProjectionOptions,
) {
    const preflight = preflightJsonInput(input, options.json_input_limits);
    if (!preflight.success) {
        throw new ConversationValidationError(
            'Transcript projection input failed JSON preflight',
            preflight.diagnostics,
        );
    }
    if (!isRecord(input) || !Array.isArray(input.turns)) {
        return validateShape(
            ConversationTranscriptProjectionInputSchema,
            input,
            'Transcript projection input schema validation failed',
        );
    }
    const sanitizedInput = {
        ...structuredClone(input),
        turns: input.turns.map((turn) => sanitizeMissingModelArguments(turn, omittedBlocks)),
    };
    const parsed = validateShape(
        ConversationTranscriptProjectionInputSchema,
        sanitizedInput,
        'Transcript projection input schema validation failed',
    );
    return { ...parsed, turns: parsed.turns.map((turn) => parseSourceTurn(turn, omittedBlocks)) };
}

function assertTranscriptFragmentSemantics(fragment: ConversationTranscriptFragment): void {
    const turnIds = fragment.turns.map((turn) => turn.id);
    const includedBlockIds = fragment.turns.flatMap((turn) =>
        turn.blocks.flatMap((block) =>
            block.type === 'tool_result' ? [block.id, ...block.content.map((content) => content.id)] : [block.id],
        ),
    );
    const omittedTurnIds = fragment.completeness.omitted_turns.map((item) => item.turn_id);
    const omittedBlockIds = fragment.completeness.omitted_blocks.map((item) => item.block_id);
    const omittedAssetIds = fragment.completeness.omitted_assets.map((item) => item.asset_id);
    const omittedGenerationIds = fragment.completeness.omitted_generations.map((item) => item.generation_id);
    const generationIds = Object.keys(fragment.generations);
    const referencedGenerationIds = [
        ...new Set(
            fragment.turns.flatMap((turn) =>
                turn.kind === 'agent' && turn.generation_id !== undefined ? [turn.generation_id] : [],
            ),
        ),
    ];
    const assetIds = Object.keys(fragment.assets);
    assertUnique(turnIds, 'Transcript turns');
    assertUnique(includedBlockIds, 'Transcript blocks');
    assertUnique(omittedTurnIds, 'Omitted transcript turns');
    assertUnique(omittedBlockIds, 'Omitted transcript blocks');
    assertUnique(omittedAssetIds, 'Omitted transcript assets');
    assertUnique(omittedGenerationIds, 'Omitted transcript generations');
    assertUnique(generationIds, 'Transcript generations');
    assertUnique(assetIds, 'Transcript assets');
    if (Object.entries(fragment.generations).some(([key, generation]) => key !== generation.id)) {
        throw new ConversationTranscriptProjectionError(
            'invalid_fragment',
            'Transcript fragment generation record identities are inconsistent',
        );
    }
    if (Object.entries(fragment.assets).some(([key, asset]) => key !== asset.id)) {
        throw new ConversationTranscriptProjectionError(
            'invalid_fragment',
            'Transcript fragment asset record identities are inconsistent',
        );
    }
    if (
        assetIds.length > CONVERSATION_TRANSCRIPT_MAX_ASSETS ||
        generationIds.length > CONVERSATION_TRANSCRIPT_MAX_GENERATIONS ||
        omittedTurnIds.some((id) => turnIds.includes(id)) ||
        omittedBlockIds.some((id) => includedBlockIds.includes(id)) ||
        omittedAssetIds.some((id) => assetIds.includes(id)) ||
        omittedGenerationIds.some((id) => generationIds.includes(id)) ||
        generationIds.some((id) => !referencedGenerationIds.includes(id)) ||
        omittedGenerationIds.some((id) => !referencedGenerationIds.includes(id)) ||
        referencedGenerationIds.some((id) => !generationIds.includes(id) && !omittedGenerationIds.includes(id)) ||
        fragment.completeness.omitted_blocks.some((item) => !turnIds.includes(item.turn_id))
    ) {
        throw new ConversationTranscriptProjectionError(
            'invalid_fragment',
            'Transcript fragment includes inconsistent included and omitted identities',
        );
    }
    const referencedAssetIds = new Set<string>();
    const inspectMediaBlock = (block: { type: string } & Record<string, unknown>) => {
        const assetId = mediaAssetId(block);
        if (assetId === undefined) return;
        const asset = Object.hasOwn(fragment.assets, assetId) ? fragment.assets[assetId] : undefined;
        if (asset?.kind !== block.type) {
            throw new ConversationTranscriptProjectionError(
                'invalid_fragment',
                'Transcript fragment asset references are inconsistent',
            );
        }
        referencedAssetIds.add(assetId);
    };
    for (const turn of fragment.turns) {
        for (const block of turn.blocks) {
            if (block.type === 'tool_result') {
                for (const content of block.content) {
                    inspectMediaBlock(content as { type: string } & Record<string, unknown>);
                }
            } else {
                inspectMediaBlock(block as { type: string } & Record<string, unknown>);
            }
        }
    }
    if (referencedAssetIds.size !== assetIds.length) {
        throw new ConversationTranscriptProjectionError(
            'invalid_fragment',
            'Transcript fragment contains an unreferenced asset',
        );
    }
    const projectedMetadata = fragment.turns.some((turn) => turn.metadata !== undefined);
    if ((fragment.completeness.metadata === 'partial') !== projectedMetadata) {
        throw new ConversationTranscriptProjectionError(
            'invalid_fragment',
            'Transcript metadata completeness differs from its projection',
        );
    }
    const semanticPartial =
        fragment.completeness.gap_before ||
        fragment.completeness.gap_after ||
        fragment.completeness.omitted_blocks.length > 0 ||
        fragment.completeness.omitted_generations.length > 0 ||
        fragment.completeness.omitted_assets.length > 0 ||
        fragment.completeness.omitted_turns.some((item) => item.reason !== 'internal_program');
    if ((fragment.completeness.semantic_content === 'partial') !== semanticPartial) {
        throw new ConversationTranscriptProjectionError(
            'invalid_fragment',
            'Transcript fragment completeness does not match its omissions and gaps',
        );
    }
}

export function parseConversationTranscriptFragment(
    input: unknown,
    options: ConversationTranscriptProjectionOptions = {},
): ConversationTranscriptFragment {
    const preflight = preflightJsonInput(input, options.json_input_limits);
    if (!preflight.success) {
        throw new ConversationValidationError('Transcript fragment failed JSON preflight', preflight.diagnostics);
    }
    const fragment = validateShape(
        ConversationTranscriptFragmentSchema,
        input,
        'Transcript fragment schema validation failed',
    );
    assertTranscriptFragmentSemantics(fragment);
    return structuredClone(input) as ConversationTranscriptFragment;
}

/**
 * Project a bounded, already selected canonical window into the safe transcript view. Storage and API
 * callers remain responsible for pinning the source revision and selecting the bounded window.
 */
export function createConversationTranscriptFragment(
    input: unknown,
    options: ConversationTranscriptProjectionOptions = {},
): ConversationTranscriptFragment {
    const omittedBlocks: ConversationTranscriptBlockOmission[] = [];
    const parsed = parseProjectionInput(input, omittedBlocks, options);
    const turnIds = parsed.turns.map((turn) => turn.id);
    const inputOmittedTurnIds = parsed.window.omitted_turns.map((item) => item.turn_id);
    const inputOmittedGenerationIds = parsed.window.omitted_generations.map((item) => item.generation_id);
    const generationIds = parsed.generations.map((generation) => generation.id);
    const assetIds = parsed.assets.map((asset) => asset.id);
    assertUnique(turnIds, 'Transcript source turns');
    assertUnique(inputOmittedTurnIds, 'Transcript source omissions');
    assertUnique(inputOmittedGenerationIds, 'Transcript source generation omissions');
    assertUnique(generationIds, 'Transcript source generations');
    assertUnique(assetIds, 'Transcript source assets');
    if (inputOmittedGenerationIds.some((id) => generationIds.includes(id))) {
        throw new ConversationTranscriptProjectionError(
            'duplicate_reference',
            'A transcript source generation cannot also be declared omitted',
        );
    }
    if (inputOmittedTurnIds.some((id) => turnIds.includes(id))) {
        throw new ConversationTranscriptProjectionError(
            'duplicate_reference',
            'A transcript source turn cannot also be declared omitted',
        );
    }

    const generationsById = new Map(parsed.generations.map((generation) => [generation.id, generation]));
    const assetsById = new Map(parsed.assets.map((asset) => [asset.id, asset]));
    const includedAssetIds = new Set<string>();
    const omittedAssets: ConversationTranscriptAssetOmission[] = [];
    const omittedTurns: ConversationTranscriptTurnOmission[] = parsed.window.omitted_turns.map((item) => ({ ...item }));
    const turns: ConversationTranscriptTurn[] = [];
    for (const turn of parsed.turns) {
        if (turn.kind === 'program' && turn.presentation !== 'transcript') {
            appendBounded(omittedTurns, { turn_id: turn.id, reason: 'internal_program' }, 'omitted turns');
            continue;
        }
        const projected = projectTurn(turn, assetsById, includedAssetIds, omittedBlocks, omittedAssets);
        const metadataKeys = options.select_turn_metadata_keys?.(Object.freeze({ id: turn.id, kind: turn.kind })) ?? [];
        if (
            metadataKeys.length > 1_024 ||
            new Set(metadataKeys).size !== metadataKeys.length ||
            metadataKeys.some((key) => !key || key === '__proto__' || key === 'constructor' || key === 'prototype')
        ) {
            throw new ConversationTranscriptProjectionError(
                'invalid_fragment',
                'Transcript metadata selector is invalid',
            );
        }
        const selectedMetadata = Object.fromEntries(
            metadataKeys.flatMap((key) =>
                turn.metadata !== undefined && Object.hasOwn(turn.metadata, key)
                    ? [[key, structuredClone(turn.metadata[key])]]
                    : [],
            ),
        );
        if (Object.keys(selectedMetadata).length) projected.metadata = selectedMetadata;
        turns.push(projected);
    }

    const referencedGenerationIds = [
        ...new Set(
            turns.flatMap((turn) =>
                turn.kind === 'agent' && turn.generation_id !== undefined ? [turn.generation_id] : [],
            ),
        ),
    ];
    if (generationIds.some((id) => !referencedGenerationIds.includes(id))) {
        throw new ConversationTranscriptProjectionError(
            'invalid_fragment',
            'Transcript source contains an unreferenced generation',
        );
    }
    if (inputOmittedGenerationIds.some((id) => !referencedGenerationIds.includes(id))) {
        throw new ConversationTranscriptProjectionError(
            'invalid_fragment',
            'Transcript source contains an unreferenced generation omission',
        );
    }
    const omittedGenerationReasons = new Map(
        parsed.window.omitted_generations.map((item) => [item.generation_id, item.reason]),
    );
    const omittedGenerations: ConversationTranscriptGenerationOmission[] = [];
    const generations = Object.fromEntries(
        referencedGenerationIds.flatMap((generationId) => {
            const generation = generationsById.get(generationId);
            if (generation !== undefined) return [[generationId, projectGeneration(generation)] as const];
            appendBounded(
                omittedGenerations,
                {
                    generation_id: generationId,
                    reason: omittedGenerationReasons.get(generationId) ?? 'not_supplied',
                },
                'omitted generations',
            );
            return [];
        }),
    );

    const assets = Object.fromEntries(
        [...includedAssetIds].map((assetId) => {
            const asset = assetsById.get(assetId);
            if (asset === undefined) {
                throw new ConversationTranscriptProjectionError(
                    'invalid_fragment',
                    'A projected transcript asset disappeared during projection',
                );
            }
            return [assetId, projectAsset(asset)];
        }),
    );
    const semanticPartial =
        parsed.window.gap_before ||
        parsed.window.gap_after ||
        parsed.window.omitted_turns.length > 0 ||
        omittedBlocks.length > 0 ||
        omittedGenerations.length > 0 ||
        omittedAssets.length > 0;
    const fragment = {
        format: CONVERSATION_TRANSCRIPT_FORMAT,
        schema_version: CONVERSATION_SCHEMA_VERSION,
        experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
        source: structuredClone(parsed.source),
        turns,
        generations,
        assets,
        completeness: {
            gap_before: parsed.window.gap_before,
            gap_after: parsed.window.gap_after,
            semantic_content: semanticPartial ? ('partial' as const) : ('complete' as const),
            metadata: turns.some((turn) => turn.metadata !== undefined) ? ('partial' as const) : ('omitted' as const),
            provenance: 'omitted' as const,
            native_replay: 'omitted' as const,
            omitted_turns: omittedTurns,
            omitted_generations: omittedGenerations,
            omitted_blocks: omittedBlocks,
            omitted_assets: omittedAssets,
        },
    };
    const outputPreflight = preflightJsonInput(fragment, options.json_input_limits);
    if (!outputPreflight.success) {
        throw new ConversationValidationError(
            'Transcript fragment exceeded the canonical JSON limit',
            outputPreflight.diagnostics,
        );
    }
    return parseConversationTranscriptFragment(fragment, options);
}
