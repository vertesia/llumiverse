import { defineConfig } from 'vitest/config';
import { testLogging } from '../vitest-logging.mjs';

export default defineConfig({
    test: {
        ...testLogging,
        // Several files each migrate and verify 100k immutable records. Keep these
        // CPU/memory-heavy fixtures from competing with each other in one package.
        // Explicit concurrent operations inside each test remain concurrent.
        fileParallelism: false,
    },
});
