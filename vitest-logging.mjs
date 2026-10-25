// Shared by every Vitest entrypoint in this repository. Console output is shown only for failing tests;
// set LLUMIVERSE_TEST_LOGS=all to see the output of passing tests too.
export function getTestLogging(env = process.env) {
    return { silent: env.LLUMIVERSE_TEST_LOGS === 'all' ? false : 'passed-only' };
}

export const testLogging = getTestLogging();
