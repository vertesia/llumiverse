import { Base64DataSource } from '@llumiverse/core';
import { describe, expect, test } from 'vitest';
import { dataSourceToVertexSourceData, isGcsUrl } from './source-utils.js';

describe('isGcsUrl', () => {
    test('recognises gs:// scheme', () => {
        expect(isGcsUrl('gs://my-bucket/file.jpg')).toBe(true);
    });

    test('recognises storage.googleapis.com URL', () => {
        expect(isGcsUrl('https://storage.googleapis.com/my-bucket/file.jpg')).toBe(true);
    });

    test('recognises storage.cloud.google.com URL', () => {
        expect(isGcsUrl('https://storage.cloud.google.com/bucket/obj')).toBe(true);
    });

    test('rejects arbitrary https URL', () => {
        expect(isGcsUrl('https://example.com/file.jpg')).toBe(false);
    });

    test('rejects S3 URL', () => {
        expect(isGcsUrl('s3://my-bucket/file.mp4')).toBe(false);
    });
});

describe('dataSourceToVertexSourceData', () => {
    test('returns gcsUri for gs:// URL', async () => {
        const ds = new Base64DataSource('file.mp4', 'video/mp4', 'abc');
        // Simulate a GCS URL by overriding getURL
        ds.getURL = async () => 'gs://my-bucket/video.mp4';
        const result = await dataSourceToVertexSourceData(ds);
        expect(result.gcsUri).toBe('gs://my-bucket/video.mp4');
        expect(result.bytesBase64Encoded).toBeUndefined();
    });

    test('returns base64 bytes for Base64DataSource with non-GCS URL', async () => {
        const b64 = Buffer.from('hello').toString('base64');
        const ds = new Base64DataSource('img.jpg', 'image/jpeg', b64);
        const result = await dataSourceToVertexSourceData(ds);
        expect(result.bytesBase64Encoded).toBe(b64);
        expect(result.gcsUri).toBeUndefined();
    });

    test('falls back to base64 when getURL rejects', async () => {
        const b64 = Buffer.from('world').toString('base64');
        const ds = new Base64DataSource('img.jpg', 'image/jpeg', b64);
        ds.getURL = async () => {
            throw new Error('no url');
        };
        const result = await dataSourceToVertexSourceData(ds);
        expect(result.bytesBase64Encoded).toBe(b64);
    });
});
