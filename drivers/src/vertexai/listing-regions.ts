/** Keep global availability plus one regional listing: configured region, otherwise a US fallback. */
export function selectVertexListingRegions(regions: readonly string[], configuredRegion: string): string[] {
    const regional =
        configuredRegion !== 'global' && regions.includes(configuredRegion)
            ? configuredRegion
            : regions.find((region) => region === 'us' || region.startsWith('us-'));
    return regions.filter((region) => region === 'global' || region === regional);
}
