import type { ToolDefinition as LegacyToolDefinition } from '@llumiverse/common';
import {
    type ToolDefinition as CanonicalToolDefinition,
    ConversationValidationError,
    deriveConversationId,
    fingerprintJson,
    type JsonObject,
    preflightJsonInput,
} from '@llumiverse/conversation';

interface CanonicalToolDefinitionSnapshot {
    name: string;
    description?: string;
    input_schema: JsonObject | boolean;
}

function snapshotToolDefinitions(
    tools: readonly LegacyToolDefinition[] | undefined,
): CanonicalToolDefinitionSnapshot[] {
    const projected = Array.from(tools ?? [], (tool): CanonicalToolDefinitionSnapshot => {
        const name = tool.name;
        const description = tool.description;
        const inputSchema = tool.input_schema as JsonObject | boolean;
        return {
            name,
            ...(description === undefined ? {} : { description }),
            input_schema: inputSchema,
        };
    });
    const preflight = preflightJsonInput(projected);
    if (!preflight.success) {
        // Preserve the historical validation contract from fingerprintJson while validating the
        // complete catalog before asynchronous hashing can observe caller mutations.
        throw new ConversationValidationError('Fingerprint input failed JSON preflight', preflight.diagnostics);
    }
    return projected.map((tool) => structuredClone(tool));
}

/** Convert an ordered legacy model-tool catalog into canonical, fingerprinted definitions. */
export async function canonicalToolDefinitions(
    tools: readonly LegacyToolDefinition[] | undefined,
): Promise<CanonicalToolDefinition[]> {
    const snapshots = snapshotToolDefinitions(tools);
    const definitions: CanonicalToolDefinition[] = [];
    for (const tool of snapshots) {
        const versionHash = await fingerprintJson({
            name: tool.name,
            description: tool.description ?? null,
            input_schema: tool.input_schema,
        });
        definitions.push({
            id: await deriveConversationId('tool_definition', tool.name, versionHash),
            name: tool.name,
            version: versionHash,
            ...(tool.description === undefined ? {} : { description: tool.description }),
            input_schema: tool.input_schema,
        });
    }
    return definitions;
}
