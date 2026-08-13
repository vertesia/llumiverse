import { z } from 'zod';

// Runtime schema for `HttpTimeoutOptions`, the per-run upstream timeouts an interaction execution
// configuration can carry. It lives here rather than in `@vertesia/common` because the type is
// declared here — restating it one package out would be the copy this migration removes.
//
// It is a LEAF of the `Project` closure: `InteractionExecutionConfiguration.http_timeout` references
// it, and a canonical component may not `$ref` a TypeScript-derived one, so this has to convert
// before that does. Like `ModelOptions` next door, publishing it from this package puts an ordering
// constraint on the release — `@llumiverse/common` ships before the studio code that reads it.
//
// `//` rather than `/** */`: a JSDoc block immediately preceding an exported declaration is picked up
// by Vertesia's OpenAPI scanner and published as that component's `description`.

// `strictObject` for the reason `model-options.ts` gives: the derived component has always said
// `additionalProperties: false`, and plain `z.object` would have PARSED an unknown key by dropping it.
//
// The `description` reproduces what the scanner already derived from the interface's TSDoc, including
// the run of spaces where it collapsed the defaults list onto one line. Reproducing the contract, not
// renegotiating it — the published document has to stay byte-identical through this conversion.
export const HttpTimeoutOptionsSchema = z
    .strictObject({
        headersTimeout: z
            .number()
            .optional()
            .meta({ description: 'Time (ms) to wait for the first response byte after the request is sent.' }),
        bodyTimeout: z
            .number()
            .optional()
            .meta({ description: 'Time (ms) between body chunks once streaming has started.' }),
        connectTimeout: z.number().optional().meta({ description: 'TCP/TLS connect timeout (ms).' }),
        keepAliveTimeout: z.number().optional().meta({ description: 'Idle socket reuse timeout (ms).' }),
    })
    .meta({
        id: 'HttpTimeoutOptions',
        description:
            "HTTP timeouts applied to a driver's upstream LLM-provider calls.\n\nAll values are in " +
            'milliseconds. Drivers should map these onto whatever HTTP client their SDK uses; the defaults ' +
            'applied in `@llumiverse/core/createDriverHttpAgent` are:   - headersTimeout:   900_000   - ' +
            'bodyTimeout:      900_000   - connectTimeout:   10_000   - keepAliveTimeout: 30_000\n\nThe ' +
            'response defaults are deliberately longer than the hosting request boundary. Application-level ' +
            'cancellation should end user work first; driver timeouts are bounded-resource safety nets.',
    });
