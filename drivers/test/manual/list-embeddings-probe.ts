/**
 * Manual probe — verifies what each provider's list-models API surface returns for
 * embedding models. This is not part of the vitest suite; it exists to back the
 * decision to drive `listEmbeddingModels()` from a static catalog rather than
 * intersecting with the provider API (since the API returns ids but no capability
 * fields such as dimensions or supported modalities).
 *
 * Usage:
 *   cd composableai/llumiverse/drivers
 *   pnpm tsx test/manual/list-embeddings-probe.ts
 *
 * Required env vars (each probe is skipped if its key is missing):
 *   OPENAI_API_KEY
 *   BEDROCK_REGION (and standard AWS credentials in the environment)
 *   MISTRAL_API_KEY
 *   GOOGLE_PROJECT_ID + GOOGLE_REGION
 */

async function probeOpenAI() {
    if (!process.env.OPENAI_API_KEY) {
        console.log('[openai] skipped — set OPENAI_API_KEY');
        return;
    }
    const { default: OpenAI } = await import('openai');
    const c = new OpenAI();
    const list = await c.models.list();
    const embed = list.data.filter((m) => m.id.includes('embed'));
    console.log('[openai] embedding ids:', embed.map((m) => m.id));
    console.log('[openai] sample fields:', embed[0] ? Object.keys(embed[0]) : '<none>');
}

async function probeBedrock() {
    if (!process.env.BEDROCK_REGION) {
        console.log('[bedrock] skipped — set BEDROCK_REGION');
        return;
    }
    const { Bedrock, ListFoundationModelsCommand } = await import('@aws-sdk/client-bedrock');
    const c = new Bedrock({ region: process.env.BEDROCK_REGION });
    const r = await c.send(new ListFoundationModelsCommand({ byOutputModality: 'EMBEDDING' }));
    console.log(
        '[bedrock] embedding foundation models:',
        r.modelSummaries?.map((m) => ({
            id: m.modelId,
            input: m.inputModalities,
            output: m.outputModalities,
            customizable: m.customizationsSupported,
            inference: m.inferenceTypesSupported,
            // Note: no `dimensions` field is returned by this API.
        })),
    );
}

async function probeMistral() {
    if (!process.env.MISTRAL_API_KEY) {
        console.log('[mistral] skipped — set MISTRAL_API_KEY');
        return;
    }
    const resp = await fetch('https://api.mistral.ai/v1/models', {
        headers: { authorization: `Bearer ${process.env.MISTRAL_API_KEY}` },
    });
    const json = (await resp.json()) as { data: Array<{ id: string; owned_by?: string }> };
    const embed = json.data.filter((m) => m.id.includes('embed'));
    console.log('[mistral] embedding ids:', embed.map((m) => m.id));
    console.log('[mistral] sample fields:', embed[0] ? Object.keys(embed[0]) : '<none>');
}

async function main() {
    console.log('Probing provider list-models endpoints for embedding metadata coverage...\n');
    await probeOpenAI().catch((e) => console.error('[openai] error:', e));
    await probeBedrock().catch((e) => console.error('[bedrock] error:', e));
    await probeMistral().catch((e) => console.error('[mistral] error:', e));
    console.log('\nDone. Note the absence of dimension/MRL fields — that gap is why the static');
    console.log('catalog in `common/src/options/embedding.ts` is the source of truth for capabilities.');
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
