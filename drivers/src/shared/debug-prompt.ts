export const BINARY_TRUNCATED_MARKER = '...[truncated binary]...';

const BINARY_DEBUG_EDGE_CHARS = 10;

export function truncateBinaryForDebug(value: string): string {
    if (value.length <= BINARY_DEBUG_EDGE_CHARS * 2 + BINARY_TRUNCATED_MARKER.length) {
        return value;
    }
    return `${value.slice(0, BINARY_DEBUG_EDGE_CHARS)}${BINARY_TRUNCATED_MARKER}${value.slice(-BINARY_DEBUG_EDGE_CHARS)}`;
}

export function truncateDataUrlForDebug(value: string): string {
    const base64Marker = ';base64,';
    const markerIndex = value.indexOf(base64Marker);
    if (!value.startsWith('data:') || markerIndex === -1) {
        return value;
    }
    const payloadStart = markerIndex + base64Marker.length;
    return value.slice(0, payloadStart) + truncateBinaryForDebug(value.slice(payloadStart));
}

export function uint8ArrayToBase64ForDebug(bytes: Uint8Array): string {
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}
