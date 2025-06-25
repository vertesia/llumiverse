
export async function readStreamAsBase64(stream: ReadableStream): Promise<string> {
    return (await _readStreamAsBuffer(stream)).toString('base64');
}

export async function readStreamAsString(stream: ReadableStream): Promise<string> {
    return (await _readStreamAsBuffer(stream)).toString();
}

export async function readStreamAsUint8Array(stream: ReadableStream): Promise<Uint8Array> {
    // We return a Uint8Array for strict type checking, even though the buffer extends Uint8Array.
    const buffer = await _readStreamAsBuffer(stream);
    return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

async function _readStreamAsBuffer(stream: ReadableStream): Promise<Buffer> {
    const out: Buffer[] = [];
    for await (const chunk of stream) {
        out.push(Buffer.from(chunk));
    }
    return Buffer.concat(out);
}
