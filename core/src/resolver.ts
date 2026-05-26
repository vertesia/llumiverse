/**
 * Get the property named by "name" of the given object
 * If an array is indexed using a string key then a map is done and an array with the content of the properties with that name are returned
 * Ex: docs.text => will return an array of text properties of the docs array
 * @param object the object
 * @param name the name of the property.
 * @returns the property value
 */
function _prop(object: unknown, name: string): unknown {
    if (object === undefined || object === null) {
        return undefined;
    }
    if (Array.isArray(object)) {
        const index = +name;
        if (Number.isNaN(index)) {
            // map array to property
            return object.map(item => item && typeof item === 'object' ? (item as Record<string, unknown>)[name] : undefined);
        } else {
            return object[index];
        }
    } else if (typeof object === 'object') {
        return (object as Record<string, unknown>)[name];
    } else {
        return undefined;
    }

}

export function resolveField(object: unknown, path: string[]): unknown {
    let p: unknown = object;
    if (!p) return p;
    if (!path.length) return p;
    const last = path.length - 1;
    for (let i = 0; i < last; i++) {
        p = _prop(p, path[i])
        if (!p) {
            return undefined;
        }
    }
    return _prop(p, path[last]);
}
