export interface BatchBucketLocation {
    bucket: string;
    prefix: string;
}

export function parseBatchBucketLocation(spec: string, scheme: 'gs' | 's3'): BatchBucketLocation {
    let value = spec.trim();
    const schemePrefix = `${scheme}://`;
    if (value.startsWith(schemePrefix)) {
        value = value.slice(schemePrefix.length);
    }

    const slash = value.indexOf('/');
    if (slash === -1) {
        return { bucket: value, prefix: '' };
    }

    let prefixEnd = value.length;
    while (prefixEnd > slash + 1 && value[prefixEnd - 1] === '/') {
        prefixEnd--;
    }
    return {
        bucket: value.slice(0, slash),
        prefix: value.slice(slash + 1, prefixEnd),
    };
}
