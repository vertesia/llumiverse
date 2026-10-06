/** Only explicit unsupported indexed adapter features use this error. Invalid source,
 * custody, ownership, schema and unexpected compiler failures retain their original errors. */
export class IndexedNativeCapabilityUnavailable extends TypeError {
    constructor(message: string, cause?: Error) {
        super(message, { cause });
        this.name = 'IndexedNativeCapabilityUnavailable';
    }
}
