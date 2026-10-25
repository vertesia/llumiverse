import { readFileSync } from 'node:fs';
import type { JSONSchema } from '@llumiverse/core';
import { Base64DataSource, PromptRole, type PromptSegment } from '@llumiverse/core';
import type { NovaMessagesPrompt } from '@llumiverse/core/formatters';

// Tabby cat photo committed with the tests so the image tests never depend on a remote host (Wikimedia rate-limits
// CI). "Domestic Short Hair Cat.jpg" by Benjamin Carr, CC0 1.0, via Wikimedia Commons:
// https://commons.wikimedia.org/wiki/File:Domestic_Short_Hair_Cat.jpg
// 466x786 JPEG — within Bedrock's [320, 4096] dimension range.
const CAT_IMAGE = readFileSync(new URL('./tabby_cat.jpg', import.meta.url)).toString('base64');

export const testPrompt_color: PromptSegment[] = [
    {
        role: PromptRole.user,
        content: 'What color is the sky?',
    },
];

export const testSchema_color: JSONSchema = {
    type: 'object',
    properties: {
        color: {
            type: 'string',
        },
    },
    required: ['color'],
    additionalProperties: false,
};

export const testPrompt_describeImage: PromptSegment[] = [
    {
        content:
            'You are a lab assistant analysing images of animals, then tag the images with accurate description of the animal shown in the picture.',
        role: PromptRole.user,
        files: [new Base64DataSource('tabby_cat.jpg', 'image/jpeg', CAT_IMAGE)],
    },
];

export const testSchema_animalDescription: JSONSchema = {
    type: 'object',
    properties: {
        name: {
            type: 'string',
        },
        type: {
            type: 'string',
        },
        species: {
            type: 'string',
        },
        characteristics: {
            type: 'array',
            items: {
                type: 'string',
            },
        },
    },
};

export const testPrompt_textToImage: NovaMessagesPrompt = {
    messages: [
        {
            role: PromptRole.user,
            content: [
                {
                    text: 'A blue sky with a purple unicorn flying',
                },
            ],
        },
    ],
};
