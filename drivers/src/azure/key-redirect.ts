/** Headers that carry an Azure resource key: `api-key` on the OpenAI APIs, `x-api-key` on Claude Messages. */
const KEY_HEADERS = ['api-key', 'x-api-key'];

/**
 * Wrap a fetch so that requests carrying an Azure resource key do not follow redirects. Fetch forwards custom headers
 * across origins on a redirect, so following one could send the key to another host. The legacy `AzureOpenAI` client
 * makes the same choice for `api-key` requests; a redirect then surfaces as an error instead of being followed.
 * https://fetch.spec.whatwg.org/#http-redirect-fetch
 */
export function withoutKeyRedirects(fetchImpl: typeof fetch): typeof fetch {
    return (input, init) => {
        const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined));
        if (KEY_HEADERS.some((name) => headers.has(name))) {
            return fetchImpl(input, { ...init, redirect: 'manual' });
        }
        return fetchImpl(input, init);
    };
}
