import { JSONValue } from "@llumiverse/common";

function extractJsonFromText(text: string): string {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    text = text.substring(start, end + 1);
    return text.replace(/\\n/g, "");
}

export function extractAndParseJSON(text: string): JSONValue {
    return parseJSON(extractJsonFromText(text));
}

const RX_DQUOTE = /^"([^"\\]|\\.)*"/us;
const RX_SQUOTE = /^'([^'\\]|\\.)*'/us;
const RX_NUMBER = /^-?(Infinity|\d+(\.\d+)?([eE][+-]?\d+)?)/;
const RX_BOOLEAN = /^(true|false|True|False)/i;
const RX_NULL = /^(null|None|undefined)/i;
const RX_KEY = /^[$_a-zA-Z][$_a-zA-Z0-9]*/;
const RX_PUNCTUATION = /^\s*([\[\]{}:,])\s*/;
const RX_COMMENT = /^(\/\/[^\n]*|\/\*[\s\S]*?\*\/)\s*/;

function fixText(value: string) {
    return value.replaceAll('\n', '\\n').replaceAll('\r', '\\r');
}

function decodeSingleQuotedString(value: string) {
    return JSON.parse('"' + value.slice(1, -1).replaceAll(/(?<!\\)"/g, '\\"') + '"');
}

export class JsonParser {
    pos: number = 0;

    constructor(public text: string) { }

    skip(n: number) {
        this.text = this.text.substring(n);
        this.pos += n;
    }

    skipComments() {
        while (true) {
            const m = RX_COMMENT.exec(this.text);
            if (m) {
                this.skip(m[0].length);
            } else {
                break;
            }
        }
    }

    tryReadPunctuation() {
        this.skipComments();
        const m = RX_PUNCTUATION.exec(this.text);
        if (m) {
            this.skip(m[0].length);
            this.skipComments();
            return m[1];
        }
    }

    readKey() {
        this.skipComments();
        const first = this.text.charCodeAt(0);
        if (first === 34) { // "
            const m = RX_DQUOTE.exec(this.text);
            if (m) {
                this.skip(m[0].length);
                this.skipComments();
                return JSON.parse(m[0]);
            }
        } else if (first === 39) { // '
            const m = RX_SQUOTE.exec(this.text);
            if (m) {
                this.skip(m[0].length);
                this.skipComments();
                return decodeSingleQuotedString(m[0]);
            }
        } else {
            const m = RX_KEY.exec(this.text);
            if (m) {
                this.skip(m[0].length);
                this.skipComments();
                return m[0];
            }
        }
        throw new Error('Expected a key at position ' + this.pos + ' but found ' + this.text);
    }

    readScalar() {
        this.skipComments();
        const first = this.text.charCodeAt(0);
        if (first === 34) { // "
            const m = RX_DQUOTE.exec(this.text);
            if (m) {
                this.skip(m[0].length);
                return JSON.parse(fixText(m[0]));
            }
        } else if (first === 39) { // '
            const m = RX_SQUOTE.exec(this.text);
            if (m) {
                this.skip(m[0].length);
                return decodeSingleQuotedString(fixText(m[0]));
            }
        } else {
            let m = RX_NUMBER.exec(this.text);
            if (m) {
                this.skip(m[0].length);
                const numStr = m[0];
                if (numStr === 'Infinity' || numStr === '-Infinity') {
                    return numStr === 'Infinity' ? Infinity : -Infinity;
                }
                const num = parseFloat(numStr);
                return isNaN(num) ? null : num; // Convert NaN to null for JSON compatibility
            }
            m = RX_BOOLEAN.exec(this.text);
            if (m) {
                this.skip(m[0].length);
                // Normalize Python-style True/False to lowercase
                return m[0].toLowerCase() === 'true';
            }
            m = RX_NULL.exec(this.text);
            if (m) {
                this.skip(m[0].length);
                // Normalize None/undefined to null
                return null;
            }
        }
        throw new Error('Expected a value at position ' + this.pos + ' but found ' + this.text);
    }

    readObject() {
        let key: string | undefined;
        const obj: any = {};
        while (true) {
            if (!key) { // read key
                const p = this.tryReadPunctuation();
                if (p === '}') {
                    return obj;
                } else if (p === ',') {
                    // Allow trailing comma before }
                    continue;
                } else if (p) {
                    throw new Error('Expected a key at position ' + this.pos + ' but found ' + this.text);
                }
                key = this.readKey();
                if (!key) {
                    throw new Error('Expected a key at position ' + this.pos + ' but found ' + this.text);
                }
                if (this.tryReadPunctuation() !== ':') {
                    throw new Error('Expected a colon at position ' + this.pos + ' but found ' + this.text);
                };
            } else { // read value
                const value = this.readValue();
                if (value === undefined) {
                    throw new Error('Expected a value at position ' + this.pos + ' but found ' + this.text);
                }
                obj[key] = value;
                key = undefined;
            }
        }
    }

    readArray() {
        const ar: any[] = [];
        while (true) {
            const p = this.tryReadPunctuation();
            if (p === ',') {
                continue;
            } else if (p === ']') {
                return ar;
            } else if (p === '[') {
                ar.push(this.readArray());
            } else if (p === '{') {
                ar.push(this.readObject());
            } else if (!p) {
                ar.push(this.readScalar());
            } else {
                throw new Error('Expected a value at position ' + this.pos + ' but found ' + this.text);
            }
        }
    }

    readValue() {
        const p = this.tryReadPunctuation();
        if (p === '{') {
            return this.readObject();
        } else if (p === '[') {
            return this.readArray();
        } else if (!p) {
            return this.readScalar();
        }
    }

    static parse(text: string) {
        const parser = new JsonParser(text);
        const r = parser.readValue();
        if (r === undefined) {
            throw new Error('Not a valid JSON');
        }
        return r;
    }
}


export function parseJSON(text: string): JSONValue {
    text = text.trim();
    try {
        return JSON.parse(text);
    } catch (err: any) {
        // use a relaxed parser
        try {
            return JsonParser.parse(text);
        } catch (err2: any) { // throw the original error
            throw err;
        }
    }
}
