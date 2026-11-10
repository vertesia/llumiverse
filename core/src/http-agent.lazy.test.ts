import { describe, expect, it, vi } from 'vitest';

// Any module-level `import ... from 'undici'` resolves through Vitest's module graph and so
// through this mock; the lazy `require` in http-agent.ts does not.
vi.mock('undici', () => {
    throw new Error('undici was imported at module load');
});

describe('http-agent', () => {
    it('does not load undici when the driver base class is imported', async () => {
        const { AbstractDriver } = await import('./Driver.js');
        expect(AbstractDriver).toBeTypeOf('function');
    });

    it('loads undici on first use', async () => {
        const { createDriverHttpAgent } = await import('./http-agent.js');
        const agent = createDriverHttpAgent();
        expect(agent.constructor.name).toBe('Agent');
        await agent.close();
    });
});
