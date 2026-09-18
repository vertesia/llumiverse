# Llumiverse common

Shared types, enums and model option metadata for Llumiverse clients and drivers.

## Adding or changing model options

`src/schemas/model-options.ts` is the canonical wire contract. Keep its
`ModelOptionsSchema` union updated whenever a factory emits a new `_option_id`.
Defining or exporting a branch schema alone does not register it in the union.

1. Add a `z.strictObject` schema with a unique, required `_option_id` literal and a
   stable `.meta({ id: 'ProviderOptions' })` component name. Append it to the union
   to preserve existing generated-client branch order.
2. Derive the public option type with `z.infer<typeof ProviderOptionsSchema>` using
   type-only imports. Keep runtime schemas on the `/schemas` subpath.
3. Return `ModelOptionsInfo` from option metadata factories and route through
   `getOptions()`. Its `_option_id` is `ModelOptions['_option_id']`: an unregistered
   ID fails compilation. Do not widen it to `string` or bypass it with a cast.
4. Add factory/routing and schema tests for representative valid options, invalid
   values and unknown fields. Metadata may be model-dependent; the wire schema
   must cover supported fields without imposing one model's limits on all models.
   Extend `src/options/options-contract.test.ts` with representative routing boundaries.
   It checks defaults, values, enum choices, field types, and conditional controls
   against the selected schema, and derives required family coverage from the union.
   Use `numeric_list` for numeric arrays, not `string_list`. The provider switch is
   exhaustive: select a factory or explicitly choose the generic fallback for every
   new provider. Retained legacy schemas need a documented compatibility test.
5. Run `pnpm lint`, `pnpm build`, `pnpm typecheck:test`, and `pnpm test` in `common`,
   then `pnpm build` at the repository root. Consumers publishing OpenAPI must
   regenerate their schema artifacts and verify discriminator mappings and request
   validation. Publish the updated common package before consumers require it.

Tests derive component membership from the union rather than repeating an ID list.
That checks emission, not factory completeness; the typed factory return contract
provides the independent completeness check. The negative compile-time regression
assertion must run through `typecheck:test`, since Vitest does not typecheck.

Retain this schema union and typed factory boundary for now. A combined provider
registry would add runtime coupling to Zod and still needs to represent factories
that select several option sets by model. Reconsider it if routing and metadata
registration develop further independent lists; any registry must preserve the
schema subpath boundary and many-to-many provider/option relationships.
