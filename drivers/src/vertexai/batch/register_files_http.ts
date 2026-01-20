    /**
 * Minimal fetch-based helper that replicates google.genai.Files.register_files().
 * Drop this file into your own TypeScript project and wire in your auth flow.
 */

declare const require: any | undefined;
declare const module: any | undefined;

const DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta';

export interface RegisterFilesOptions {
    /** OAuth2 access token that can call the Gemini Developer file service. */
    accessToken: string;
    /** Cloud project to bill; becomes the x-goog-user-project header. */
    quotaProject?: string;
    /** Storage URIs (gs://bucket/object) to register. */
    uris: string[];
    /** Override base URL (handy for testing/emulators). */
    baseUrl?: string;
    /** Custom fetch implementation for SSR or tests. */
    fetchImpl?: typeof fetch;
}

export interface FileResource {
    name: string;
    uri?: string;
    mimeType?: string;
    displayName?: string;
    downloadUri?: string;
    source?: string;
    sizeBytes?: string;
    createTime?: string;
    updateTime?: string;
}

export interface RegisterFilesResponse {
    files: FileResource[];
    /** Raw JSON payload in case you need additional fields. */
    raw: unknown;
    /** Native Response so callers can inspect headers/status. */
    httpResponse: Response;
}

export async function registerFilesViaHttp(options: RegisterFilesOptions): Promise<RegisterFilesResponse> {
    const { accessToken, quotaProject, uris, baseUrl = DEFAULT_BASE_URL, fetchImpl } = options;
    if (!Array.isArray(uris) || !uris.length) {
        throw new Error('uris must be a non-empty array of storage URIs.');
    }

    const fetchFn = fetchImpl ?? globalThis.fetch;
    if (!fetchFn) {
        throw new Error('A fetch implementation must be provided in non-browser environments.');
    }

    const url = `${baseUrl.replace(/\/$/, '')}/files:register`;
    const headers: Record<string, string> = {
        'content-type': 'application/json',
        authorization: `Bearer ${accessToken}`,
    };
    if (quotaProject) {
        headers['x-goog-user-project'] = quotaProject;
    }

    const payload = { uris };

    const response = await fetchFn(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const errorBody = await safeReadJson(response);
        const message = (errorBody && (errorBody.error?.message ?? errorBody.message)) || response.statusText;
        throw new Error(`files:register failed (${response.status}): ${message}`);
    }

    const json = await response.json();
    return {
        files: Array.isArray(json?.files) ? json.files : [],
        raw: json,
        httpResponse: response,
    };
}

async function safeReadJson(response: Response): Promise<any> {
    try {
        return await response.clone().json();
    } catch (_err) {
        return undefined;
    }
}

/**
 * Example usage. Replace the placeholders with real values or remove this block.
 */
async function example() {
    const globalProcess = (globalThis as any)?.process;
    const accessToken = globalProcess?.env?.GOOGLE_ACCESS_TOKEN ?? '<replace-with-oauth-token>';
    const quotaProject = globalProcess?.env?.GOOGLE_QUOTA_PROJECT;
    const uris = ['gs://my-bucket/path/to/object.txt'];

    const result = await registerFilesViaHttp({ accessToken, quotaProject, uris });
    console.log('Registered files:', result.files.map((file) => file.name));
}

const isExecutedDirectly =
    typeof require !== 'undefined' &&
    typeof module !== 'undefined' &&
    require.main === module;

if (isExecutedDirectly) {
    example().catch((err) => {
        console.error(err);
        const globalProcess = (globalThis as any)?.process;
        if (globalProcess) {
            globalProcess.exitCode = 1;
        }
    });
}
