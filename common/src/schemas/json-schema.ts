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
//    unconstrained `{}` here, and a canonical component must stay byte-identical to the one the
//    scanner derives for the slots that have not converted. The INTERFACE still states the union, so
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
    })
    .meta({ id: 'JSONSchema' });

// A property map, written as an object with a catchall rather than `z.record`: `z.record` also emits
// a `propertyNames: {type: 'string'}` — a no-op, since every JSON object key is a string, but not
// what the index signature published, and these components are pinned byte-identical.
export const JSONSchemaPropertiesSchema: z.ZodType<JSONSchemaProperties> = z
    .object({})
    .catchall(JSONSchemaSchema)
    .meta({ id: 'JSONSchemaProperties' });

// NOT a JSDoc block: this file's types are published as OpenAPI components, and the scanner emits a
// leading `/** */` as the component's `description`. Rationale goes in `//` comments so it stays in
// the source instead of shipping to every client generator.
//
// Written as an interface rather than as `z.infer<typeof JSONSchemaSchema>`, for two reasons, of
// which only the second is a workaround:
//
//  1. A RECURSIVE Zod schema cannot infer its own type — the getters make the inference circular —
//     so `z.ZodType<...>` has to be handed a named result. The interface therefore exists either
//     way; `z.infer` would only change which name is the public one, not remove a declaration. What
//     prevents drift is the annotation: add a property to one side and not the other and
//     `z.ZodType<JSONSchema>` stops accepting the schema, at compile time, in this file.
//  2. Even where it WOULD remove a declaration, the OpenAPI scanner cannot follow it yet.
//     `ts-json-schema-generator` derives components from the TypeScript AST, and a `z.infer<>` alias
//     resolves to nothing. Aliasing these two emptied 141 of the 1015 published components — every
//     closure reaching `JSONSchema` through a component not yet converted — and collapsed eight more
//     to closed empty objects. Until every referrer publishes a canonical component, a public type
//     reachable from a derived one has to stay a declaration the scanner can read.
export interface JSONSchema {
    type?: JSONSchemaTypeName | JSONSchemaTypeName[];
    description?: string;
    properties?: JSONSchemaProperties;
    items?: JSONSchema;
    format?: string;
    editor?: unknown;
    default?: unknown;
    additionalProperties?: boolean | JSONSchema;
    required?: string[];
    [k: string]: unknown;
}

// A property map. Same reasoning as JSONSchema above for why it is an interface.
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
