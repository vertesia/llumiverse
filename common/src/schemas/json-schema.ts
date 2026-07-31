import { z } from 'zod';

// Runtime schemas for the JSON Schema subset llumiverse passes to the drivers.
//
// Vertesia publishes both as OpenAPI components and validates request bodies against the SAME schema
// object at runtime, which is what makes the published contract and the enforced one incapable of
// disagreeing. The interfaces below carry the same shape because a recursive schema has to be handed
// a named type — see the note above `JSONSchema` — and `z.ZodType<JSONSchema>` is what stops the two
// from drifting: they are checked against each other by the compiler, in this file, on every build.
//
// `//` rather than `/** */` throughout: a JSDoc block immediately preceding an exported declaration
// is picked up by Vertesia's OpenAPI scanner and published as that component's `description`.
//
// Two shape decisions, both deliberate:
//
//  - The object is OPEN — `looseObject`, not `object`. A JSON Schema carries keywords this type never
//    enumerated (`enum`, `oneOf`, `minimum`, `$ref`, …) and callers do send them. `z.object` would
//    not merely publish an open component, it would STRIP those keywords from anything it parses:
//    `JSONSchemaSchema.parse({type: 'string', enum: [...]})` returns `{type: 'string'}`. For a type
//    whose whole purpose is to carry a user's schema, silently deleting half of it is the worst
//    available failure, and it is invisible — the parse succeeds.
//  - `type` is `z.any()`, not the `JSONSchemaTypeName | JSONSchemaTypeName[]` union it morally is —
//    the one place in this file `any` is right. The published component has always emitted the
//    unconstrained `{}` here, and a canonical component must still agree with the one the scanner
//    derives for the slots that have not converted. The NAMED TYPE still states the union, so
//    TypeScript callers get the real type; only the emitted schema is loose. Tightening it is a
//    contract change, possible once nothing derives these — at which point it is a one-line edit.
export const JSONSchemaSchema: z.ZodType<JSONSchema> = z
    .looseObject({
        type: z.any().optional(),
        description: z.string().optional(),
        get properties() {
            return JSONSchemaPropertiesSchema.optional();
        },
        get items() {
            return JSONSchemaSchema.optional();
        },
        format: z.string().optional(),
        editor: z.unknown().optional(),
        default: z.unknown().optional(),
        get additionalProperties() {
            return z.union([z.boolean(), JSONSchemaSchema]).optional();
        },
        required: z.array(z.string()).optional(),
    } satisfies KnownFieldSchemas)
    .meta({ id: 'JSONSchema' });

// A property map, written as an object with a catchall rather than `z.record`: `z.record` also emits
// a `propertyNames: {type: 'string'}` — a no-op, since every JSON object key is a string, but not
// what the index signature published, and these components still have to agree with what the
// scanner derives.
export const JSONSchemaPropertiesSchema: z.ZodType<JSONSchemaProperties> = z
    .object({})
    .catchall(JSONSchemaSchema)
    .meta({ id: 'JSONSchemaProperties' });

// NOT a JSDoc block: this file's types are published as OpenAPI components, and the scanner emits a
// leading `/** */` as the component's `description`. Rationale goes in `//` comments so it stays in
// the source instead of shipping to every client generator.
//
// Every other type in this closure is now `z.infer` of its schema. This one is not, and the reason
// is RECURSION, not the scanner: the scanner short-circuits a `z.infer` alias to the published
// component rather than deriving it, so that obstacle is gone. What remains is TypeScript's own
// limit — Zod 4 infers a recursive type from the getters below, but the inference bottoms out at
// depth, so `items` degrades to `{}` a few levels down and driver code that walks a nested schema
// stops compiling. A recursive schema has to be handed a named type.
//
// The exception is recorded, with this reason, in `packages/api-specs/canonical-aliases.json`, and
// the gate there rejects it if it ever becomes inferable — so it cannot quietly outlive the
// constraint. Because this type is NOT short-circuited, the scanner still derives it wherever an
// unconverted type reaches it, and the derived result must still agree with the canonical component.
//
// The known fields are split out from the index signature so the named type can actually be CHECKED
// against the schema. `z.ZodType<JSONSchema>` alone does not check much: every named field is
// optional and the type is open, so `z.looseObject({})` — a schema declaring nothing at all —
// satisfies it. What catches a field present on one side and missing from the other is
// {@link KnownFieldSchemas} below, which requires the Zod shape to cover every key of
// `JSONSchemaKnownFields` exactly.
interface JSONSchemaKnownFields {
    type?: JSONSchemaTypeName | JSONSchemaTypeName[];
    description?: string;
    properties?: JSONSchemaProperties;
    items?: JSONSchema;
    format?: string;
    editor?: unknown;
    default?: unknown;
    additionalProperties?: boolean | JSONSchema;
    required?: string[];
}

// The extras a JSON Schema carries that this type never enumerated — `enum`, `oneOf`, `minimum`,
// `$ref`. They are why the object is open, and why `looseObject` rather than `object` is load-bearing
// rather than cosmetic: Zod's default STRIPS them on parse.
export type JSONSchema = JSONSchemaKnownFields & Record<string, unknown>;

// `-?` is what makes this exact rather than a subset check: an optional key in the mapped type would
// let a missing schema property pass. With it, dropping `description` from the shape above fails to
// compile here rather than silently narrowing the published component.
type KnownFieldSchemas = {
    [K in keyof Required<JSONSchemaKnownFields>]-?: z.ZodType<JSONSchemaKnownFields[K]>;
};

// A property map. Recursive through `JSONSchema`, so it keeps a named type for the same reason.
export interface JSONSchemaProperties {
    [key: string]: JSONSchema;
}

export type JSONSchemaTypeName =
    | 'string' //
    | 'number'
    | 'integer'
    | 'boolean'
    | 'object'
    | 'array'
    | 'null'
    | 'any';
