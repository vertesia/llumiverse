import { type ExecutionOptions, isClaudeVersionGTE, type JSONSchema } from '@llumiverse/core';

/**
 * Native Claude structured outputs (`output_config.format`) on Amazon Bedrock.
 *
 * Bedrock constrains Claude's answer to a JSON schema, but it accepts only a subset of JSON Schema and returns a 400
 * for anything else. A request uses the native format only when the model and the schema both qualify; every other
 * request keeps the schema instructions in the prompt, as before.
 * https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html
 */

/** Bedrock rejects a schema with more optional properties than this, counted across nested objects. */
const MAX_OPTIONAL_PROPERTIES = 12;
/** Bedrock rejects a schema with more union-typed properties (type arrays or `anyOf`) than this. */
const MAX_UNION_PROPERTIES = 16;

/** Keywords Bedrock does not support for Claude structured outputs. */
const UNSUPPORTED_KEYWORDS = [
    'minimum',
    'maximum',
    'exclusiveMinimum',
    'exclusiveMaximum',
    'multipleOf',
    'minLength',
    'maxLength',
    // Not in Bedrock's supported list, so a schema using them keeps the prompt path.
    'oneOf',
    'not',
    'if',
    'then',
    'else',
    'patternProperties',
    'dependentSchemas',
    'unevaluatedProperties',
    'prefixItems',
] as const;

/** Claude 4.5 and later accept structured outputs on Bedrock. */
export function supportsClaudeNativeResultSchema(model: string): boolean {
    return isClaudeVersionGTE(model, 4, 5);
}

/**
 * Whether a Claude request on Bedrock sends its result schema as `output_config.format` instead of prompt text.
 * Requests with tools keep the prompt path: agent tool loops carry the schema on every turn, and how Bedrock combines
 * a constrained answer with tool calls is not documented.
 */
export function usesClaudeNativeResultSchema(
    options: Pick<ExecutionOptions, 'model' | 'result_schema' | 'tools'>,
): boolean {
    return (
        options.result_schema !== undefined &&
        !options.tools?.length &&
        supportsClaudeNativeResultSchema(options.model) &&
        isClaudeStructuredOutputSchema(options.result_schema)
    );
}

interface SchemaStats {
    optionalProperties: number;
    unionProperties: number;
}

type SchemaNode = Record<string, unknown>;

function isSchemaNode(value: unknown): value is SchemaNode {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Whether a JSON schema stays within what Bedrock accepts for Claude structured outputs: an object at the root,
 * `additionalProperties: false` on every object, none of the unsupported keywords, `minItems` of 0 or 1, primitive
 * `enum` values, only internal non-recursive `$ref`s, and the optional-property and union limits.
 */
export function isClaudeStructuredOutputSchema(schema: JSONSchema): boolean {
    if (!isSchemaNode(schema) || schema.type !== 'object') return false;
    const definitions = { ...asNode(schema.$defs, '$defs'), ...asNode(schema.definitions, 'definitions') };
    if (hasRecursiveReference(schema, definitions)) return false;
    const stats: SchemaStats = { optionalProperties: 0, unionProperties: 0 };
    return (
        checkNode(schema, stats) &&
        stats.optionalProperties <= MAX_OPTIONAL_PROPERTIES &&
        stats.unionProperties <= MAX_UNION_PROPERTIES
    );
}

/** Definitions keyed by their `$ref` path, e.g. `#/$defs/Address`. */
function asNode(value: unknown, container: string): Record<string, unknown> {
    if (!isSchemaNode(value)) return {};
    return Object.fromEntries(Object.entries(value).map(([name, node]) => [`#/${container}/${name}`, node]));
}

function checkNode(node: unknown, stats: SchemaStats): boolean {
    if (!isSchemaNode(node)) return false;
    if (UNSUPPORTED_KEYWORDS.some((keyword) => keyword in node)) return false;
    if ('minItems' in node && node.minItems !== 0 && node.minItems !== 1) return false;
    if ('enum' in node) {
        if (!Array.isArray(node.enum)) return false;
        if (!node.enum.every((value) => value === null || ['string', 'number', 'boolean'].includes(typeof value))) {
            return false;
        }
    }
    if (typeof node.$ref === 'string' && !node.$ref.startsWith('#/')) return false;

    const isObject = node.type === 'object' || (Array.isArray(node.type) && node.type.includes('object'));
    if (isObject || 'properties' in node) {
        if (node.additionalProperties !== false) return false;
        const properties = isSchemaNode(node.properties) ? node.properties : {};
        const required = Array.isArray(node.required) ? node.required : [];
        for (const [name, property] of Object.entries(properties)) {
            if (!required.includes(name)) stats.optionalProperties++;
            if (isSchemaNode(property) && (Array.isArray(property.type) || 'anyOf' in property)) {
                stats.unionProperties++;
            }
            if (!checkNode(property, stats)) return false;
        }
    }

    if ('items' in node && !checkNode(node.items, stats)) return false;
    for (const keyword of ['anyOf', 'allOf'] as const) {
        if (keyword in node) {
            const branches = node[keyword];
            if (!Array.isArray(branches) || !branches.every((branch) => checkNode(branch, stats))) return false;
        }
    }
    for (const container of ['$defs', 'definitions'] as const) {
        if (container in node) {
            const definitions = node[container];
            if (!isSchemaNode(definitions) || !Object.values(definitions).every((def) => checkNode(def, stats))) {
                return false;
            }
        }
    }
    return true;
}

/** Collects every `$ref` string under a node. */
function collectReferences(node: unknown, references: Set<string>): Set<string> {
    if (Array.isArray(node)) {
        for (const item of node) collectReferences(item, references);
    } else if (isSchemaNode(node)) {
        for (const [key, value] of Object.entries(node)) {
            if (key === '$ref' && typeof value === 'string') references.add(value);
            else collectReferences(value, references);
        }
    }
    return references;
}

/** A reference to the root, to an unknown target, or a cycle between definitions makes the schema recursive. */
function hasRecursiveReference(schema: SchemaNode, definitions: Record<string, unknown>): boolean {
    const rootReferences = collectReferences(schema, new Set());
    if ([...rootReferences].some((ref) => ref === '#' || (ref.startsWith('#/') && !(ref in definitions)))) {
        return true;
    }
    const visiting = new Set<string>();
    const done = new Set<string>();
    const visit = (ref: string): boolean => {
        if (done.has(ref)) return false;
        if (visiting.has(ref)) return true;
        visiting.add(ref);
        for (const next of collectReferences(definitions[ref], new Set())) {
            if (next === '#' || visit(next)) return true;
        }
        visiting.delete(ref);
        done.add(ref);
        return false;
    };
    return Object.keys(definitions).some(visit);
}
