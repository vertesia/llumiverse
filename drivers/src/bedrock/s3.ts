import { CreateBucketCommand, HeadBucketCommand, S3Client } from "@aws-sdk/client-s3";
import { Progress, Upload } from "@aws-sdk/lib-storage";

export async function doesBucketExist(s3: S3Client, bucketName: string): Promise<boolean> {
    try {
        await s3.send(new HeadBucketCommand({ Bucket: bucketName }));
        return true;
    } catch (err: any) {
        if (err.name === 'NotFound') {
            return false;
        }
        throw err;
    }
}

export function createBucket(s3: S3Client, bucketName: string) {
    return s3.send(new CreateBucketCommand({
        Bucket: bucketName
    }));
}


export async function tryCreateBucket(s3: S3Client, bucketName: string) {
    const exists = await doesBucketExist(s3, bucketName);
    if (!exists) {
        return createBucket(s3, bucketName);
    }
}


export async function uploadFile(s3: S3Client, source: ReadableStream, bucketName: string, file: string, onProgress?: (progress: Progress) => void) {

    const upload = new Upload({
        client: s3,
        params: {
            Bucket: bucketName,
            Key: file,
            Body: source,
        }
    });

    onProgress && upload.on("httpUploadProgress", onProgress);

    const result = await upload.done();
    return result;
}

/**
 * Create the bucket if not already exists and then upload the file.
 * @param s3 
 * @param source 
 * @param bucketName 
 * @param file 
 * @param onProgress 
 * @returns 
 */
export async function forceUploadFile(s3: S3Client, source: ReadableStream, bucketName: string, file: string, onProgress?: (progress: Progress) => void) {
    // make sure the bucket exists
    await tryCreateBucket(s3, bucketName);
    return uploadFile(s3, source, bucketName, file, onProgress);
}


/**
 * Parse an S3 HTTPS URL into an S3 URI format
 * s3Url - The S3 HTTPS URL (e.g., https://bucket.s3.region.amazonaws.com/key)
 * returns The S3 URI (e.g., s3://bucket/key)
 */
export function parseS3UrlToUri(s3Url: URL) {
    try {
        const url = new URL(s3Url);

        // Extract the hostname which contains the bucket and S3 endpoint
        const hostname = url.hostname;

        // Parse the hostname to extract the bucket name
        let bucketName;
        if (hostname.includes('.s3.')) {
            // Format: bucket-name.s3.region.amazonaws.com
            bucketName = hostname.split('.s3.')[0];
        } else if (hostname.startsWith('s3.') && hostname.includes('.amazonaws.com')) {
            // Format: s3.region.amazonaws.com/bucket-name
            // In this case, the bucket is actually in the first segment of the pathname
            bucketName = url.pathname.split('/')[1];
            // Adjust the pathname to remove the bucket name
            const pathParts = url.pathname.split('/').slice(2);
            url.pathname = '/' + pathParts.join('/');
        } else {
            throw new Error('Unable to determine bucket name from URL');
        }

        // The key is the pathname without the leading slash
        // If we had the bucket name in the path, it's already been removed above
        let key = url.pathname;
        if (key.startsWith('/')) {
            key = key.substring(1);
        }

        // Construct the S3 URI
        return `s3://${bucketName}/${key}`;
    } catch (error) {
        console.error('Error parsing S3 URL:', error);
        throw error;
    }
}