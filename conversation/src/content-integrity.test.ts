import { describe, expect, it } from 'vitest';
import {
    canonicalJsonContentString,
    hashCanonicalJsonContent,
    hashContentBytes,
    hashUtf8Content,
    inlineAssetContentIntegrity,
} from './content-integrity.js';

describe('canonical content integrity', () => {
    it('hashes exact binary bytes with an independently computed SHA-256', async () => {
        await expect(hashContentBytes(new Uint8Array([0, 1, 255]))).resolves.toEqual({
            content_hash: 'sha256:26a66b061e8f48f39927c312f25293959729eee95978e2892d49d3512a5cc092',
            byte_length: 3,
        });
        await expect(inlineAssetContentIntegrity({ type: 'inline_base64', data: 'AAH/' })).resolves.toEqual({
            content_hash: 'sha256:26a66b061e8f48f39927c312f25293959729eee95978e2892d49d3512a5cc092',
            byte_length: 3,
        });
    });

    it('hashes text as UTF-8 bytes', async () => {
        await expect(hashUtf8Content('café')).resolves.toEqual({
            content_hash: 'sha256:850f7dc43910ff890f8879c0ed26fe697c93a067ad93a7d50f466a7028a9bf4e',
            byte_length: 5,
        });
        await expect(hashUtf8Content('😀')).resolves.toEqual({
            content_hash: 'sha256:f0443a342c5ef54783a111b51ba56c938e474c32324d90c3a60c9c8e3a37e2d9',
            byte_length: 4,
        });
        await expect(hashUtf8Content('\ud800')).rejects.toThrow(/unpaired surrogate/);
        await expect(hashUtf8Content('\udc00')).rejects.toThrow(/unpaired surrogate/);
    });

    it('hashes inline JSON using documented stable sorted-key UTF-8 bytes', async () => {
        expect(canonicalJsonContentString({ b: 2, a: 1 })).toBe('{"a":1,"b":2}');
        await expect(hashCanonicalJsonContent({ b: 2, a: 1 })).resolves.toEqual({
            content_hash: 'sha256:43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777',
            byte_length: 13,
        });
        await expect(hashCanonicalJsonContent({ a: 1, b: 2 })).resolves.toEqual(
            await hashCanonicalJsonContent({ b: 2, a: 1 }),
        );
        expect(canonicalJsonContentString({ value: '\ud800' })).toBe('{"value":"\\ud800"}');
    });

    it('does not claim byte integrity for external locators and rejects malformed base64', async () => {
        await expect(
            inlineAssetContentIntegrity({ type: 'external', resolver: 'url', locator: { url: 'https://x.test/a' } }),
        ).resolves.toBeUndefined();
        await expect(inlineAssetContentIntegrity({ type: 'inline_base64', data: 'not base64' })).rejects.toThrow(
            /valid base64/,
        );
    });
});
