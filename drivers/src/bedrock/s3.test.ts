import { describe, expect, it } from 'vitest';
import { isAmazonS3Hostname, parseS3UrlToUri } from './s3.js';

describe('isAmazonS3Hostname', () => {
    it('accepts supported Amazon S3 hostnames', () => {
        expect(isAmazonS3Hostname('s3.amazonaws.com')).toBe(true);
        expect(isAmazonS3Hostname('s3.us-east-1.amazonaws.com')).toBe(true);
        expect(isAmazonS3Hostname('s3.dualstack.us-east-1.amazonaws.com')).toBe(true);
        expect(isAmazonS3Hostname('my-bucket.s3.amazonaws.com')).toBe(true);
        expect(isAmazonS3Hostname('my-bucket.s3.us-east-1.amazonaws.com')).toBe(true);
        expect(isAmazonS3Hostname('my-bucket.s3.dualstack.us-east-1.amazonaws.com')).toBe(true);
    });

    it('rejects hostnames that only contain amazonaws.com as a substring', () => {
        expect(isAmazonS3Hostname('s3.amazonaws.com.example.com')).toBe(false);
        expect(isAmazonS3Hostname('example-s3.amazonaws.com')).toBe(false);
        expect(isAmazonS3Hostname('bucket.s3.amazonaws.com.evil.test')).toBe(false);
    });
});

describe('parseS3UrlToUri', () => {
    it('converts virtual-hosted style S3 URLs', () => {
        expect(parseS3UrlToUri(new URL('https://my-bucket.s3.us-east-1.amazonaws.com/folder/video.mp4'))).toBe(
            's3://my-bucket/folder/video.mp4',
        );
    });

    it('converts path-style S3 URLs', () => {
        expect(parseS3UrlToUri(new URL('https://s3.us-east-1.amazonaws.com/my-bucket/folder/video.mp4'))).toBe(
            's3://my-bucket/folder/video.mp4',
        );
    });

    it('rejects spoofed S3-looking URLs', () => {
        expect(() => parseS3UrlToUri(new URL('https://bucket.s3.amazonaws.com.evil.test/video.mp4'))).toThrow(
            'Unable to determine bucket name from URL',
        );
    });
});
