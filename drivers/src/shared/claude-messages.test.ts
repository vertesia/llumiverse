import { PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { formatClaudePrompt } from './claude-messages.js';

describe('formatClaudePrompt', () => {
    it('warns and skips video attachments', async () => {
        const getStream = vi.fn();
        const warn = vi.fn();
        const segments = [
            {
                role: PromptRole.user,
                content: 'Look at this',
                files: [
                    {
                        name: 'clip.mp4',
                        mime_type: 'video/mp4',
                        getStream,
                    },
                ],
            },
        ] as unknown as PromptSegment[];

        const prompt = await formatClaudePrompt(segments, { model: 'claude-haiku-4-5' } as never, { warn });

        expect(prompt.messages).toEqual([
            {
                role: 'user',
                content: [{ type: 'text', text: 'Look at this' }],
            },
        ]);
        expect(getStream).not.toHaveBeenCalled();
        expect(warn).toHaveBeenCalledWith(
            { file_name: 'clip.mp4', mime_type: 'video/mp4' },
            '[Claude] Skipping unsupported video attachment',
        );
    });
});
