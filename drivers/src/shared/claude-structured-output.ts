import type { MessageParam } from '@anthropic-ai/sdk/resources/index.js';
import type { CompletionChunkObject, DriverCompletionStream } from '@llumiverse/common';
import { type ExecutionOptions, isClaudeVersionGTE, type JSONSchema } from '@llumiverse/core';

/**
 * Native Claude structured outputs (`output_config.format`) on Amazon Bedrock and Google Vertex AI.
 *
 * Both platforms constrain Claude's answer to a JSON schema, but they accept only a subset of JSON Schema and return a
 * 400 for anything else. A request uses the native format only when the model and the schema both qualify; every
 * other request keeps the schema instructions in the prompt, as before.
 * https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html
 * https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/structured-outputs
 */

/** Providers that receive the result schema as `output_config.format` when the request qualifies. */
export const CLAUDE_NATIVE_RESULT_SCHEMA_PROVIDERS: ReadonlySet<string> = new Set(['bedrock', 'vertexai']);

/** The platforms reject a schema with more optional properties than this, counted across nested objects. */
const MAX_OPTIONAL_PROPERTIES = 12;
/** The platforms reject a schema with more union-typed properties (type arrays or `anyOf`) than this. */
const MAX_UNION_PROPERTIES = 16;

/** Keywords the platforms do not support for Claude structured outputs. */
const UNSUPPORTED_KEYWORDS = [
    'minimum',
    'maximum',
    'exclusiveMinimum',
    'exclusiveMaximum',
    'multipleOf',
    'minLength',
    'maxLength',
    // Array constraints beyond `minItems` of 0 or 1.
    'maxItems',
    'uniqueItems',
    'contains',
    'minContains',
    'maxContains',
    'additionalItems',
    // Object constraints other than `required` and `additionalProperties: false`.
    'minProperties',
    'maxProperties',
    'propertyNames',
    'dependentRequired',
    'dependencies',
    // Not in the supported list, so a schema using them keeps the prompt path.
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

/** The only string `format` values the platforms accept for Claude structured outputs. */
const SUPPORTED_STRING_FORMATS: ReadonlySet<unknown> = new Set([
    'date-time',
    'time',
    'date',
    'duration',
    'email',
    'hostname',
    'uri',
    'ipv4',
    'ipv6',
    'uuid',
]);

/** Regex features the platforms reject in `pattern`: backreferences, lookaround assertions and word boundaries. */
const UNSUPPORTED_PATTERN = /\\[1-9]|\\k<|\(\?<?[=!]|\\[bB]/;

/** Claude 4.5 and later accept structured outputs. */
export function supportsClaudeNativeResultSchema(model: string): boolean {
    return isClaudeVersionGTE(model, 4, 5);
}

/**
 * Whether a Claude request sends its result schema as `output_config.format` instead of prompt text, on the
 * providers in {@link CLAUDE_NATIVE_RESULT_SCHEMA_PROVIDERS}. Requests with tools keep the prompt path: agent tool
 * loops carry the schema on every turn, and how a constrained answer combines with tool calls is not documented.
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
 * Whether a JSON schema stays within what the platforms accept for Claude structured outputs: an object at the root,
 * `additionalProperties: false` on every object, none of the unsupported keywords, `minItems` of 0 or 1, supported
 * string formats, primitive `enum` values, only internal non-recursive `$ref`s, and the optional-property and union
 * limits.
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
    if ('format' in node && !SUPPORTED_STRING_FORMATS.has(node.format)) return false;
    if ('pattern' in node && (typeof node.pattern !== 'string' || UNSUPPORTED_PATTERN.test(node.pattern))) return false;
    // `allOf` does not support `$ref` branches.
    if (Array.isArray(node.allOf) && node.allOf.some((branch) => isSchemaNode(branch) && '$ref' in branch)) {
        return false;
    }

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

/**
 * Whether a request ends with an assistant turn, which Claude continues as a prefill. Structured outputs reject a
 * prefill, so such a request keeps the schema instructions in the prompt.
 */
export function endsWithClaudePrefill(messages: readonly MessageParam[]): boolean {
    return messages.findLast((message) => (message.role as string) !== 'system')?.role === 'assistant';
}

/**
 * Whether an error is Vertex AI refusing structured outputs because an organization policy
 * (`constraints/vertexai.allowedPartnerModelFeatures`) does not allow the `structured_outputs` feature for the model.
 * Vertex AI answers 400 FAILED_PRECONDITION; any other 400 is a different problem and is not matched.
 * https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/structured-outputs#control_access_to_structured_outputs
 */
export function isClaudeStructuredOutputPolicyDenial(error: unknown): boolean {
    if (typeof error !== 'object' || error === null) return false;
    const { status, message, error: body } = error as { status?: unknown; message?: unknown; error?: unknown };
    if (status !== 400) return false;
    let text = typeof message === 'string' ? message : '';
    if (body !== undefined) {
        try {
            text += ` ${JSON.stringify(body)}`;
        } catch {
            // An unserializable body adds nothing to match.
        }
    }
    return (
        text.includes('FAILED_PRECONDITION') &&
        text.includes('allowedPartnerModelFeatures') &&
        text.includes('structured_outputs')
    );
}

/** At most this many denied project, region and model combinations are remembered; the oldest is dropped first. */
const MAX_REMEMBERED_DENIALS = 256;

/**
 * Project, region and model combinations whose structured outputs a policy denied, so later requests in this
 * process go straight to the prompt path instead of failing once each.
 */
export class ClaudeStructuredOutputDenials {
    private readonly denied = new Set<string>();

    static key(project: string | undefined, region: string, model: string): string {
        return `${project ?? ''}|${region}|${model}`;
    }

    has(key: string): boolean {
        return this.denied.has(key);
    }

    add(key: string): void {
        this.denied.delete(key);
        this.denied.add(key);
        if (this.denied.size > MAX_REMEMBERED_DENIALS) {
            const oldest = this.denied.values().next().value;
            if (oldest !== undefined) this.denied.delete(oldest);
        }
    }
}

/**
 * A stream that runs `fallback` instead when `primary` fails with a structured-output policy denial before its first
 * chunk. A denial is a 400, so it always arrives before any chunk. The conversation finalizer follows the stream that
 * ran.
 */
export function withClaudeStructuredOutputFallback(
    primary: DriverCompletionStream,
    fallback: () => Promise<DriverCompletionStream>,
    onDenied: (error: unknown) => void,
): DriverCompletionStream {
    let active = primary;
    async function* chunks(): AsyncGenerator<CompletionChunkObject> {
        let started = false;
        try {
            for await (const chunk of primary) {
                started = true;
                yield chunk;
            }
        } catch (error) {
            if (started || !isClaudeStructuredOutputPolicyDenial(error)) throw error;
            onDenied(error);
            active = await fallback();
            yield* active;
        }
    }
    return {
        [Symbol.asyncIterator]: () => chunks(),
        finalizeConversation: () => active.finalizeConversation?.(),
        finalizePromptCacheDiagnostic: () => active.finalizePromptCacheDiagnostic?.(),
    };
}
