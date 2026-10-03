import { ConversationValidationError } from './diagnostics.js';
import { fingerprintJson } from './identity.js';
import { preflightJsonInput } from './json-preflight.js';
import { AssetSchema } from './schemas/content.js';
import type { Asset } from './types.js';

/** Fingerprints retained metadata/locator, not a claim that a declared hash matches binary content. */
export async function fingerprintAssetSelectionMetadata(assetInput: Asset): Promise<string> {
    const preflight = preflightJsonInput(assetInput);
    if (!preflight.success)
        throw new ConversationValidationError('Asset metadata failed JSON preflight', preflight.diagnostics);
    const asset = AssetSchema.parse(assetInput);
    const { storage, ...metadata } = asset;
    return fingerprintJson({ ...metadata, storage: storage.type === 'external' ? storage : { type: storage.type } });
}
