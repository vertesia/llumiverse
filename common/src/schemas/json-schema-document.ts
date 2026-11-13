import { z } from 'zod';
import * as schemas from './index.js';

/** Where the committed document lives, relative to the package root. Published as `./json-schema`. */
export const LLUMIVERSE_JSON_SCHEMA_PATH = 'json-schema/llumiverse.schema.json';

export const LLUMIVERSE_JSON_SCHEMA_ID =
    'https://github.com/vertesia/llumiverse/common/json-schema/llumiverse.schema.json';

/**
 * Builds the JSON Schema document for every Zod schema exported from `@llumiverse/common/schemas`.
 *
 * The document is COMPILED OUTPUT of the Zod schemas, not a second definition. It exists so that
 * consumers outside TypeScript (llumiverse-go generates its wire types from it) read the same shapes,
 * and `json-schema-document.test.ts` fails when the committed copy drifts from the Zod source.
 *
 * Each export becomes one entry of `$defs`, named by its registered `id` when it has one (the name the
 * Vertesia API contract already publishes) and otherwise by its export name without the `Schema`
 * suffix. References between exports are `$ref`s, so every shape is written once.
 *
 * Emitted with `io: 'input'`, matching `@vertesia/common`'s API contract: a field with a default is
 * optional to whoever writes the value.
 */
export function buildLlumiverseJsonSchema(): Record<string, unknown> {
    const registry = z.registry<{ id: string }>();
    for (const [exportName, schema] of exportedSchemas()) {
        registry.add(schema, { id: definitionName(exportName, schema) });
    }
    const { schemas: defs } = z.toJSONSchema(registry, {
        target: 'draft-2020-12',
        io: 'input',
        unrepresentable: 'any',
        uri: (id) => `#/$defs/${id}`,
    });
    const sorted: Record<string, JsonObject> = {};
    for (const name of Object.keys(defs).sort()) {
        const { $schema: _schema, $id: _id, ...definition } = defs[name] as JsonObject;
        sorted[name] = definition;
    }
    for (const definition of Object.values(sorted)) {
        // A definition root that only says "any extra key" adds nothing a consumer can type.
        if (isEmptyObject(definition.additionalProperties)) delete definition.additionalProperties;
        normalize(definition, sorted);
    }
    return {
        $schema: 'https://json-schema.org/draft/2020-12/schema',
        $id: LLUMIVERSE_JSON_SCHEMA_ID,
        title: 'Llumiverse common types',
        description:
            'Generated from the Zod schemas exported by @llumiverse/common/schemas. ' +
            'Do not edit: run `pnpm --filter @llumiverse/common gen:json-schema`.',
        $defs: sorted,
    };
}

type JsonObject = Record<string, unknown>;

const DEF_PREFIX = '#/$defs/';

function isPlainObject(value: unknown): value is JsonObject {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isEmptyObject(value: unknown): boolean {
    return isPlainObject(value) && Object.keys(value).length === 0;
}

/**
 * The shape-level adjustments `@vertesia/common`'s API-contract adapter makes to the same Zod output,
 * so a type reads the same in this document and in the published Vertesia contract:
 * - an empty `properties` is dropped;
 * - a record with unconstrained values is `additionalProperties: true`, not Zod's `{}` plus a
 *   `propertyNames: { type: 'string' }` that every JSON key already satisfies (generators otherwise
 *   type the two spellings differently);
 * - a union of objects that share a required literal property gets an OpenAPI `discriminator`.
 *
 * Strictness (`additionalProperties: false`) is left as Zod writes it: the Vertesia contract opens or
 * closes objects per request or response, which is an API policy, not a shape.
 */
function normalize(node: unknown, defs: Record<string, JsonObject>): void {
    if (Array.isArray(node)) {
        for (const child of node) normalize(child, defs);
        return;
    }
    if (!isPlainObject(node)) return;
    if (isEmptyObject(node.properties)) delete node.properties;
    if (isEmptyObject(node.additionalProperties)) {
        node.additionalProperties = true;
        if (isPlainObject(node.propertyNames) && node.propertyNames.type === 'string') {
            if (Object.keys(node.propertyNames).length === 1) delete node.propertyNames;
        }
    }
    synthesizeDiscriminator(node, defs);
    for (const child of Object.values(node)) normalize(child, defs);
}

/**
 * Adds `discriminator` to a union whose members all reference definitions sharing a REQUIRED property
 * with a distinct string literal in each. Code generators (Go, Java) need it to pick the concrete type.
 * Mirrors `synthesizeDiscriminator` in `@vertesia/common/src/api-schemas/adapter.ts`.
 */
function synthesizeDiscriminator(node: JsonObject, defs: Record<string, JsonObject>): void {
    const members = node.oneOf ?? node.anyOf;
    if (!Array.isArray(members) || members.length < 2 || node.discriminator) return;

    const targets: { ref: string; schema: JsonObject }[] = [];
    for (const member of members) {
        const ref = isPlainObject(member) ? member.$ref : undefined;
        if (typeof ref !== 'string' || !ref.startsWith(DEF_PREFIX)) return;
        const schema = defs[ref.slice(DEF_PREFIX.length)];
        if (!schema) return;
        targets.push({ ref, schema });
    }

    const candidates = new Map<string, Record<string, string>>();
    const disqualified = new Set<string>();
    for (const { ref, schema } of targets) {
        if (!isPlainObject(schema.properties)) return;
        const required = Array.isArray(schema.required) ? schema.required : [];
        for (const [key, value] of Object.entries(schema.properties)) {
            if (!isPlainObject(value)) continue;
            if (!required.includes(key)) {
                disqualified.add(key);
                continue;
            }
            const literal =
                typeof value.const === 'string'
                    ? value.const
                    : Array.isArray(value.enum) && value.enum.length === 1 && typeof value.enum[0] === 'string'
                      ? value.enum[0]
                      : undefined;
            if (literal === undefined) continue;
            const mapping = candidates.get(key) ?? {};
            if (mapping[literal]) {
                disqualified.add(key);
                continue;
            }
            mapping[literal] = ref;
            candidates.set(key, mapping);
        }
    }

    for (const [propertyName, mapping] of candidates) {
        if (disqualified.has(propertyName) || Object.keys(mapping).length !== targets.length) continue;
        node.discriminator = { propertyName, mapping };
        if (node.anyOf && !node.oneOf) {
            node.oneOf = node.anyOf;
            delete node.anyOf;
        }
        node.type = 'object';
        const required = Array.isArray(node.required) ? node.required.filter((r) => typeof r === 'string') : [];
        node.required = [...new Set([...required, propertyName])];
        return;
    }
}

function exportedSchemas(): [string, z.ZodType][] {
    const exported: [string, z.ZodType][] = [];
    for (const [name, value] of Object.entries(schemas as Record<string, unknown>)) {
        if (name.endsWith('Schema') && value instanceof z.ZodType) exported.push([name, value]);
    }
    return exported.sort(([a], [b]) => a.localeCompare(b));
}

function definitionName(exportName: string, schema: z.ZodType): string {
    const registered = z.globalRegistry.get(schema)?.id;
    return typeof registered === 'string' ? registered : exportName.replace(/Schema$/, '');
}
