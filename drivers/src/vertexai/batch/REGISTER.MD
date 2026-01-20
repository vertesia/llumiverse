# Registering Files via HTTP

The Gemini Developer file service exposes a plain HTTP endpoint so you can call it from any runtime, including browser or server-side TypeScript. This document mirrors the behavior of `google.genai.Files.register_files()` and distills it into raw HTTP primitives.

## Endpoint Summary

- **Method**: `POST`
- **URL**: `https://generativelanguage.googleapis.com/v1beta/files:register`
- **Scope**: `https://www.googleapis.com/auth/cloud-platform`

## Required Headers

- `Authorization: Bearer <ACCESS_TOKEN>` — obtain the token with your preferred Google auth flow (service account, OAuth client, etc.) and make sure it includes the Cloud Platform scope.
- `Content-Type: application/json`
- `X-Goog-User-Project: <PROJECT_ID>` *(optional but recommended when using quota project billing).*

## Request Body

Send the Cloud Storage URIs you want to register:

```json
{
  "uris": [
    "gs://my-bucket/path/to/object-one.parquet",
    "gs://my-bucket/path/to/object-two.mp3"
  ]
}
```

Each URI must point to an object that the authenticated principal can read.

## Response Shape

Successful calls return HTTP `200` with JSON that echoes back the registered files:

```json
{
  "files": [
    {
      "name": "files/abc123",
      "uri": "gs://my-bucket/path/to/object-one.parquet",
      "mimeType": "application/x-parquet",
      "displayName": "object-one.parquet",
      "source": "FILE_SOURCE_STORAGE"
    }
  ]
}
```

The server may add additional metadata fields (timestamps, size, download URIs). Preserve any fields you care about; the SDK simply forwards them to callers.

## TypeScript Helper

The sample in [samples/typescript/register_files_http.ts](samples/typescript/register_files_http.ts) packages the HTTP call into a reusable helper:

1. Load or mint an OAuth2 access token.
2. Provide the list of `gs://` URIs.
3. Optionally set `quotaProject` to match your billing project.
4. Call `registerFilesViaHttp({ accessToken, quotaProject, uris })`.

Because the helper only depends on the Fetch API, it works in browsers (with `fetch`) and in Node (with native `fetch` or `node-fetch`). Replace the placeholders in the `example()` function or import the helper into your own codebase and supply your auth headers there.

## Troubleshooting

- **403 PERMISSION_DENIED**: Ensure the token has the Cloud Platform scope and that the service account/user can read every URI.
- **404 NOT_FOUND**: Double-check the bucket/object names; the API also returns this when the storage object is missing.
- **429/503**: The file service is rate-limited. Add retries with exponential backoff (the Python SDK uses $2^n$ multipliers starting at one second).

For advanced scenarios (custom transports, tracing, alternative hosts), override the `baseUrl` option in the helper or issue the HTTP request manually with the same headers and JSON payload described above.
