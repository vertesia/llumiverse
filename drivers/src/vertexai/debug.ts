import util from 'node:util';

export function logObject(prefix: string, obj: unknown) {
    const fullObj = util.inspect(obj, { showHidden: false, depth: null, colors: true });
    console.log(prefix, fullObj);
}
