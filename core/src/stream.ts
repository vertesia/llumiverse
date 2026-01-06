
export async function readStreamAsBase64(stream: ReadableStream, maxSize?: number): Promise<string> {
    const uint8Array = await readStreamAsUint8Array(stream, maxSize);
    return Buffer.from(uint8Array).toString('base64');
}

export async function readStreamAsString(stream: ReadableStream, maxSize?: number): Promise<string> {
    const uint8Array = await readStreamAsUint8Array(stream, maxSize);
    return Buffer.from(uint8Array).toString();
}

export async function readStreamAsUint8Array(stream: ReadableStream, maxSize?: number): Promise<Uint8Array> {
    const chunks: Uint8Array[] = [];
    let totalLength = 0;
    
    for await (const chunk of stream) {
        const uint8Chunk = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
        
        // Check size before adding chunk
        totalLength += uint8Chunk.length;
        if (maxSize && totalLength > maxSize) {
            // throw new Error(
            //     `Stream size exceeds maximum allowed size of ${(maxSize / 1024 / 1024).toFixed(0)}MB. ` +
            //     `For large files (especially videos), use cloud storage instead of inline data.`
            // );
        }
        
        chunks.push(uint8Chunk);
    }
    
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
    }
    
    return combined;
}
