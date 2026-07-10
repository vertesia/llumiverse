import { describe, expect, it } from 'vitest';
import { selectLiveTestDrivers } from './live-model-selection.js';

const drivers = [
    { name: 'bedrock', models: ['bedrock-model-a', 'bedrock-model-b'] },
    { name: 'openai', models: ['openai-model-a'] },
];

describe('selectLiveTestDrivers', () => {
    it('selects only the requested provider', () => {
        expect(selectLiveTestDrivers(drivers, { providers: 'bedrock' })).toEqual([drivers[0]]);
    });

    it('selects only the requested model', () => {
        expect(selectLiveTestDrivers(drivers, { models: 'bedrock-model-b' })).toEqual([
            { name: 'bedrock', models: ['bedrock-model-b'] },
        ]);
    });

    it('rejects an unavailable provider or model', () => {
        expect(() => selectLiveTestDrivers(drivers, { providers: 'vertexai' })).toThrow(
            'No configured live test provider',
        );
        expect(() => selectLiveTestDrivers(drivers, { models: 'unknown-model' })).toThrow(
            'No configured live test model',
        );
    });
});
