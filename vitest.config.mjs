import { defineConfig } from 'vitest/config';
import { testLogging } from './vitest-logging.mjs';

export default defineConfig({
    test: {
        ...testLogging,
    },
});
