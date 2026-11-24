import { ReplayedToolNames } from '@llumiverse/core';
import type { ChatCompletionRequest } from '@mistralai/mistralai/models/components';
import { describe, expect, it } from 'vitest';
import { sanitizeMistralToolCallNames } from './index.js';

type Messages = ChatCompletionRequest['messages'];

const history = (): Messages => [
    { role: 'user', content: 'go' },
    {
        role: 'assistant',
        content: '',
        toolCalls: ['', 'functions.x', 'mcp.github.search', 'web_search'].map((name, i) => ({
            id: `c${i}`,
            function: { name, arguments: '{}' },
        })),
    },
    { role: 'tool', toolCallId: 'c0', content: 'Tool not found' },
];

describe('Mistral replayed tool names', () => {
    it('sends safe names for undeclared tools and keeps declared ones, without changing the messages', () => {
        const messages = history();
        const names = new ReplayedToolNames([{ name: 'web_search' }, { name: 'mcp.github.search' }]);

        const sent = sanitizeMistralToolCallNames(messages, names);

        const assistant = sent[1];
        expect(assistant.role === 'assistant' ? assistant.toolCalls?.map((call) => call.function.name) : []).toEqual([
            '_',
            'functions_x',
            'mcp.github.search',
            'web_search',
        ]);
        expect(messages).toEqual(history());
    });
});
