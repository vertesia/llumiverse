
function uint8ArrayToBase64(uint8Array: Uint8Array): string {
    let binary = '';

    for (const byte of uint8Array) {
        binary += String.fromCharCode(byte);
    }

    return btoa(binary);
}

export async function readStreamAsBase64(stream: ReadableStream): Promise<string> {
    const uint8Array = await readStreamAsUint8Array(stream);
    return uint8ArrayToBase64(uint8Array);
}

export async function readStreamAsString(stream: ReadableStream): Promise<string> {
    const uint8Array = await readStreamAsUint8Array(stream);
    return new TextDecoder().decode(uint8Array);
}

export async function readStreamAsUint8Array(stream: ReadableStream): Promise<Uint8Array> {
    const chunks: Uint8Array[] = [];
    let totalLength = 0;
    
    for await (const chunk of stream) {
        const uint8Chunk = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
        
        chunks.push(uint8Chunk);
        totalLength += uint8Chunk.length;
    }
    
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
    }
    
    return combined;
}
