import { z } from 'zod';
import { JSON_POINTER_PATTERN_SOURCE } from '../runtime-constants.js';
import { NonnegativeSafeIntegerSchema, PositiveSafeIntegerSchema } from './primitives.js';

export const ImageRegionSchema = z
    .strictObject({
        type: z.literal('image_region'),
        coordinate_space: z.enum(['pixels', 'normalized']),
        x: z.number().nonnegative(),
        y: z.number().nonnegative(),
        width: z.number().positive(),
        height: z.number().positive(),
    })
    .meta({ id: 'ConversationImageRegion' });

export const PageRangeSchema = z
    .strictObject({
        type: z.literal('page_range'),
        from_page: PositiveSafeIntegerSchema,
        through_page: PositiveSafeIntegerSchema,
    })
    .meta({ id: 'ConversationPageRange' });

export const TimeRangeSchema = z
    .strictObject({
        type: z.literal('time_range'),
        start_seconds: z.number().nonnegative(),
        end_seconds: z.number().positive(),
    })
    .meta({ id: 'ConversationTimeRange' });

export const TextCodePointRangeSchema = z
    .strictObject({
        start_code_point: NonnegativeSafeIntegerSchema,
        end_code_point: NonnegativeSafeIntegerSchema,
    })
    .meta({ id: 'ConversationTextCodePointRange' });
/** RFC 6901: empty root or slash-separated tokens, with only ~0 and ~1 escapes. */
export const JsonPointerSchema = z
    .string()
    .regex(new RegExp(JSON_POINTER_PATTERN_SOURCE))
    .meta({ id: 'ConversationJsonPointer' });
