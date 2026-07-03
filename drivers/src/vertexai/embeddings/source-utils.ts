import { Base64DataSource, type DataSource, dataSourceToBase64 } from '@llumiverse/core';

interface VertexSourceData {
    gcsUri?: string;
    bytesBase64Encoded?: string;
}

export function isGcsUrl(url: string): boolean {
    return (
        url.startsWith('gs://') ||
        url.startsWith('https://storage.googleapis.com/') ||
        url.startsWith('https://storage.cloud.google.com/')
    );
}

export async function dataSourceToVertexSourceData(ds: DataSource): Promise<VertexSourceData> {
    const url = await ds.getURL().catch(() => undefined);
    if (url && isGcsUrl(url)) {
        return { gcsUri: url };
    }

    const bytesBase64Encoded = ds instanceof Base64DataSource ? ds.getBase64() : await dataSourceToBase64(ds);
    if (!bytesBase64Encoded) {
        throw new Error(
            `Data source conversion produced empty result — name=${ds.name}, mime=${ds.mime_type ?? 'unknown'}`,
        );
    }
    return { bytesBase64Encoded };
}
