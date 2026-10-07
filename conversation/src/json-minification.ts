/** RFC 8259 grammar validation retaining every non-whitespace UTF-16 code unit verbatim. */
export const JSON_MINIFICATION_FORMAT = 'rfc8259-lexical-v1';

export interface JsonMinificationLimits {
    max_code_units: number;
    max_depth: number;
    max_lexical_tokens: number;
}

export const DEFAULT_JSON_MINIFICATION_LIMITS: Readonly<JsonMinificationLimits> = Object.freeze({
    max_code_units: 1024 * 1024,
    max_depth: 128,
    max_lexical_tokens: 262144,
});

export class JsonMinificationError extends Error {
    constructor(
        readonly code: 'INVALID_JSON' | 'JSON_MINIFICATION_LIMIT' | 'JSON_MINIFICATION_MISMATCH',
        readonly offset: number,
    ) {
        super(`${code} at code unit ${offset}`);
        this.name = 'JsonMinificationError';
    }
}

type Frame =
    | { kind: 'array'; state: 'value_or_end' | 'value' | 'comma_or_end' }
    | { kind: 'object'; state: 'key_or_end' | 'key' | 'colon' | 'value' | 'comma_or_end' };

const whitespace = (code: number) => code === 0x20 || code === 0x09 || code === 0x0a || code === 0x0d;
const digit = (code: number) => code >= 0x30 && code <= 0x39;
const hex = (code: number) => digit(code) || (code >= 0x41 && code <= 0x46) || (code >= 0x61 && code <= 0x66);

/** No parsing into JS values: numeric lexemes, string escapes, duplicate keys and order are retained. */
export function minifyJsonLexically(
    input: string,
    options: { limits?: Partial<JsonMinificationLimits>; signal?: AbortSignal } = {},
): string {
    if (typeof input !== 'string') throw new TypeError('JSON minification source must be a string');
    const limits = { ...DEFAULT_JSON_MINIFICATION_LIMITS, ...options.limits };
    for (const [key, value] of Object.entries(limits)) {
        const maximum = DEFAULT_JSON_MINIFICATION_LIMITS[key as keyof JsonMinificationLimits];
        if (!Number.isSafeInteger(value) || value < 1 || value > maximum)
            throw new RangeError(`Invalid JSON minification limit ${key}`);
    }
    options.signal?.throwIfAborted();
    if (input.length > limits.max_code_units) throw new JsonMinificationError('JSON_MINIFICATION_LIMIT', 0);
    let offset = 0;
    let tokens = 0;
    const pieces: string[] = [];
    const stack: Frame[] = [];
    let rootStarted = false;
    const invalid = (): never => {
        throw new JsonMinificationError('INVALID_JSON', offset);
    };
    const emit = (start: number) => {
        if (++tokens > limits.max_lexical_tokens) throw new JsonMinificationError('JSON_MINIFICATION_LIMIT', start);
        pieces.push(input.slice(start, offset));
    };
    const symbol = () => {
        const start = offset++;
        emit(start);
    };
    const string = () => {
        const start = offset++;
        for (;;) {
            if (offset >= input.length) invalid();
            const code = input.charCodeAt(offset++);
            if (code === 0x22) break;
            if (code < 0x20) invalid();
            if (code === 0x5c) {
                const escapeCode = input.charCodeAt(offset++);
                if (escapeCode === 0x75) {
                    for (let index = 0; index < 4; index++) {
                        if (!hex(input.charCodeAt(offset++))) invalid();
                    }
                } else if (![0x22, 0x5c, 0x2f, 0x62, 0x66, 0x6e, 0x72, 0x74].includes(escapeCode)) invalid();
            }
        }
        emit(start);
    };
    const value = () => {
        const start = offset;
        const code = input.charCodeAt(offset);
        if (code === 0x22) string();
        else if (code === 0x5b || code === 0x7b) {
            if (stack.length >= limits.max_depth) throw new JsonMinificationError('JSON_MINIFICATION_LIMIT', offset);
            symbol();
            stack.push(
                code === 0x5b ? { kind: 'array', state: 'value_or_end' } : { kind: 'object', state: 'key_or_end' },
            );
        } else if (code === 0x2d || digit(code)) {
            if (code === 0x2d) offset++;
            if (input.charCodeAt(offset) === 0x30) offset++;
            else {
                if (!digit(input.charCodeAt(offset)) || input.charCodeAt(offset) === 0x30) invalid();
                while (digit(input.charCodeAt(offset))) offset++;
            }
            if (input.charCodeAt(offset) === 0x2e) {
                offset++;
                if (!digit(input.charCodeAt(offset))) invalid();
                while (digit(input.charCodeAt(offset))) offset++;
            }
            if (input.charCodeAt(offset) === 0x65 || input.charCodeAt(offset) === 0x45) {
                offset++;
                if (input.charCodeAt(offset) === 0x2b || input.charCodeAt(offset) === 0x2d) offset++;
                if (!digit(input.charCodeAt(offset))) invalid();
                while (digit(input.charCodeAt(offset))) offset++;
            }
            emit(start);
        } else {
            const literal = code === 0x74 ? 'true' : code === 0x66 ? 'false' : code === 0x6e ? 'null' : '';
            if (!literal || !input.startsWith(literal, offset)) invalid();
            offset += literal.length;
            emit(start);
        }
    };
    for (;;) {
        options.signal?.throwIfAborted();
        while (whitespace(input.charCodeAt(offset))) offset++;
        const frame = stack.at(-1);
        if (!frame) {
            if (rootStarted) {
                if (offset !== input.length) invalid();
                return pieces.join('');
            }
            rootStarted = true;
            value();
        } else if (frame.kind === 'array') {
            if (frame.state === 'comma_or_end') {
                if (input[offset] === ',') {
                    symbol();
                    frame.state = 'value';
                } else if (input[offset] === ']') {
                    symbol();
                    stack.pop();
                } else invalid();
            } else if (frame.state === 'value_or_end' && input[offset] === ']') {
                symbol();
                stack.pop();
            } else {
                frame.state = 'comma_or_end';
                value();
            }
        } else if (frame.state === 'key_or_end' && input[offset] === '}') {
            symbol();
            stack.pop();
        } else if (frame.state === 'key_or_end' || frame.state === 'key') {
            if (input[offset] !== '"') invalid();
            string();
            frame.state = 'colon';
        } else if (frame.state === 'colon') {
            if (input[offset] !== ':') invalid();
            symbol();
            frame.state = 'value';
        } else if (frame.state === 'value') {
            frame.state = 'comma_or_end';
            value();
        } else if (input[offset] === ',') {
            symbol();
            frame.state = 'key';
        } else if (input[offset] === '}') {
            symbol();
            stack.pop();
        } else invalid();
    }
}

/** Core verification does not accept a plugin's claimed parser/fidelity or equivalent reserialization. */
export function assertJsonMinification(
    source: string,
    replacement: string,
    options: { limits?: Partial<JsonMinificationLimits>; signal?: AbortSignal } = {},
): void {
    if (typeof replacement !== 'string') throw new TypeError('JSON minification replacement must be a string');
    if (minifyJsonLexically(source, options) !== replacement) {
        throw new JsonMinificationError('JSON_MINIFICATION_MISMATCH', 0);
    }
}
