/**
 * Shared helpers for tests in @llumiverse/drivers. These avoid repeated `as unknown` casts
 * and centralize the patterns we need when asserting on untyped/dynamic values in tests.
 */

/**
 * Create an Error with extra typed properties (e.g. `status`, `code`).
 * Replaces the `(err as unknown).status = 429` pattern.
 *
 * @example
 *   const err = errorWith('Rate limit exceeded', { status: 429 });
 *   err.status; // typed as number
 */
export function errorWith<T extends object>(message: string, extra: T): Error & T {
    return Object.assign(new Error(message), extra);
}

/**
 * Safely read a property from an unknown value. Returns `undefined` if `obj` is null/undefined,
 * not an object, or doesn't have the key. The return type can be narrowed via the generic.
 *
 * @example
 *   const status = getProp<number>(error, 'status');
 */
export function getProp<T = unknown>(obj: unknown, key: string): T | undefined {
    if (obj != null && typeof obj === 'object' && key in obj) {
        return (obj as Record<string, unknown>)[key] as T;
    }
    return undefined;
}

/**
 * Cast an object to expose its private members for testing. The caller defines the shape
 * (typically a `type DriverInternals = { privateMethod: (...) => ReturnT; ... }`).
 *
 * @example
 *   type Internals = { isErrorRetryable: (err: unknown) => boolean };
 *   exposePrivate<Internals>(driver).isErrorRetryable(err);
 */
export function exposePrivate<T>(obj: object): T {
    return obj as unknown as T;
}

/**
 * Recursive "navigable" type for asserting on results of structure-preserving
 * transformations (e.g. stripping/serialization). Each property access returns the same
 * type, letting tests navigate deeply without per-test interfaces. For sites that call
 * string/array methods on a leaf (e.g. `.substring`), add a targeted `as string` cast.
 */
export type Tree = { readonly [key: string]: Tree };
