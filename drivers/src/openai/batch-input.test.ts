import {
    type BatchInferenceRequestItem,
    type DataSource,
    type ExecutionOptions,
    PromptRole,
    type PromptSegment,
} from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIDriver } from './openai.js';

interface BatchInputLine {
    body: {
        input: Array<{
            role?: string;
            content?: Array<{ type: string; image_url?: string }>;
        }>;
    };
}

const options: ExecutionOptions = {
    model: 'gpt-5.2',
    model_options: { _option_id: 'text-fallback', max_tokens: 100 },
};

function imageDataSource() {
    const getStream = vi.fn().mockResolvedValue(
        new ReadableStream<Uint8Array>({
            start(controller) {
                controller.enqueue(new Uint8Array([1, 2, 3]));
                controller.close();
            },
        }),
    );
    const getURL = vi.fn().mockResolvedValue('https://signed.example/image.png?signature=redacted');
    const source: DataSource = {
        name: 'image.png',
        mime_type: 'image/png',
        getStream,
        getURL,
        getURI: vi.fn().mockResolvedValue('gs://bucket/image.png'),
    };
    return { source, getStream, getURL };
}

function request(source: DataSource): BatchInferenceRequestItem {
    const segments: PromptSegment[] = [{ role: PromptRole.user, content: 'Describe the image.', files: [source] }];
    return { custom_id: 'image-1', segments, options };
}

describe('OpenAI batch image inputs', () => {
    it('places signed image URLs in the batch JSONL without reading the image bytes', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test-key' });
        const { source, getStream, getURL } = imageDataSource();
        let inputJsonl = '';
        const createFile = vi.fn().mockImplementation(async ({ file }: { file: File }) => {
            inputJsonl = await file.text();
            return { id: 'file-batch-input' };
        });
        const createBatch = vi.fn().mockResolvedValue({ id: 'batch-1', status: 'validating' });
        driver.service = {
            files: { create: createFile },
            batches: { create: createBatch },
        } as unknown as OpenAI;

        await driver.startBatchInference([request(source)], { name: 'signed-image-batch' });

        expect(getURL).toHaveBeenCalledOnce();
        expect(getStream).not.toHaveBeenCalled();
        const line = JSON.parse(inputJsonl) as BatchInputLine;
        const userMessage = line.body.input.find((item) => item.role === 'user');
        expect(userMessage?.content).toContainEqual({
            type: 'input_image',
            image_url: 'https://signed.example/image.png?signature=redacted',
            detail: 'auto',
        });
        expect(createBatch).toHaveBeenCalledWith({
            input_file_id: 'file-batch-input',
            endpoint: '/v1/responses',
            completion_window: '24h',
        });
    });

    it('keeps synchronous prompt images inline', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test-key' });
        const { source, getStream, getURL } = imageDataSource();

        const prompt = await driver.createPrompt(request(source).segments, options);

        expect(getStream).toHaveBeenCalledOnce();
        expect(getURL).not.toHaveBeenCalled();
        const userMessage = prompt.find(
            (item): item is OpenAI.Responses.EasyInputMessage => 'role' in item && item.role === 'user',
        );
        expect(userMessage?.content).toContainEqual({
            type: 'input_image',
            image_url: 'data:image/png;base64,AQID',
            detail: 'auto',
        });
    });
});
