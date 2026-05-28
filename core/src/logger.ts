import type { Logger } from '@llumiverse/common';

// Helper to create logger methods that support both message-only and object-first signatures.
function createConsoleLoggerMethod(consoleMethod: (...args: unknown[]) => void): Logger['info'] {
    return ((objOrMsg: unknown, msgOrNever?: string, ...args: (string | number | boolean)[]) => {
        if (typeof objOrMsg === 'string') {
            // Message-only: logger.info("message", ...args)
            consoleMethod(objOrMsg, msgOrNever, ...args);
        } else if (msgOrNever !== undefined) {
            // Object-first: logger.info({ obj }, "message", ...args)
            consoleMethod(msgOrNever, objOrMsg, ...args);
        } else {
            // Object-only: logger.info({ obj })
            consoleMethod(objOrMsg, ...args);
        }
    }) as Logger['info'];
}

const ConsoleLogger: Logger = {
    debug: createConsoleLoggerMethod(console.debug.bind(console)),
    info: createConsoleLoggerMethod(console.info.bind(console)),
    warn: createConsoleLoggerMethod(console.warn.bind(console)),
    error: createConsoleLoggerMethod(console.error.bind(console)),
};

const noop = () => void 0;
const NoopLogger: Logger = {
    debug: noop as Logger['debug'],
    info: noop as Logger['info'],
    warn: noop as Logger['warn'],
    error: noop as Logger['error'],
};

export function createLogger(logger: Logger | 'console' | undefined) {
    if (logger === 'console') {
        return ConsoleLogger;
    } else if (logger) {
        return logger;
    } else {
        return NoopLogger;
    }
}
