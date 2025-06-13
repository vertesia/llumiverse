
export async function readStreamAsBase64(stream: ReadableStream): Promise<string> {
    return (await _readStreamAsBuffer(stream)).toString('base64');
}

export async function readStreamAsString(stream: ReadableStream): Promise<string> {
    return (await _readStreamAsBuffer(stream)).toString();
}

export async function readStreamAsUint8Array(stream: ReadableStream): Promise<Uint8Array> {
    return _readStreamAsBuffer(stream);
}

async function _readStreamAsBuffer(stream: ReadableStream): Promise<Buffer> {
    const out: Buffer[] = [];
    for await (const chunk of stream) {
        out.push(Buffer.from(chunk));
    }
    return Buffer.concat(out);
}
