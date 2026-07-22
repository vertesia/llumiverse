export interface LiveTestDriver {
    name: string;
    models: string[];
}

export interface LiveModelSelection {
    models?: string;
    providers?: string;
}

function parseFilter(value: string | undefined): Set<string> {
    return new Set(
        value
            ?.split(',')
            .map((item) => item.trim())
            .filter(Boolean),
    );
}

export function selectLiveTestDrivers<T extends LiveTestDriver>(drivers: T[], selection: LiveModelSelection): T[] {
    const providers = parseFilter(selection.providers);
    const models = parseFilter(selection.models);
    const providerMatches = providers.size === 0 ? drivers : drivers.filter((driver) => providers.has(driver.name));

    if (providers.size > 0 && providerMatches.length === 0) {
        throw new Error(`No configured live test provider matched: ${[...providers].join(', ')}`);
    }

    const selected = providerMatches
        .map((driver) => ({
            ...driver,
            models: models.size === 0 ? driver.models : driver.models.filter((model) => models.has(model)),
        }))
        .filter((driver) => driver.models.length > 0) as T[];

    if (models.size > 0 && selected.length === 0) {
        throw new Error(`No configured live test model matched: ${[...models].join(', ')}`);
    }

    return selected;
}
