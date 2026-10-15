import { describe, expect, it } from 'vitest';
import { resolveAnthropicVertexRegion } from './claude.js';

describe('Vertex Claude region resolution', () => {
    it.each([
        ['us-central1', 'claude-sonnet-4-6', 'us-east5'],
        ['europe-west4', 'claude-haiku-4-5@20251001', 'europe-west1'],
        ['us-central1', 'claude-haiku-5-5', 'us'],
        ['europe-west4', 'claude-sonnet-5-5', 'eu'],
        ['asia-southeast1', 'claude-opus-4-7', 'global'],
        ['global', 'claude-haiku-5-5', 'global'],
    ])('maps the driver region %s for %s to %s', (driverRegion, model, expected) => {
        expect(resolveAnthropicVertexRegion(undefined, driverRegion, model)).toBe(expected);
    });

    it.each([
        ['us-east5', 'claude-haiku-5-5', 'us-east5'],
        ['us', 'claude-haiku-5-5', 'us'],
        ['eu', 'claude-sonnet-4-6', 'eu'],
        ['global', 'claude-sonnet-4-6', 'global'],
    ])('keeps the caller location %s for %s', (explicitRegion, model, expected) => {
        expect(resolveAnthropicVertexRegion(explicitRegion, 'us-central1', model)).toBe(expected);
    });
});
