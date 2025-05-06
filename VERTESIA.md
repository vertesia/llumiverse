# Build & Test Commands

- Build: `pnpm build` (builds all workspace packages)
- Package build: `cd <package-dir> && pnpm build`
- Dev mode: `pnpm dev` (watches for changes)
- Run tests: `pnpm test` (all tests)
- Run single test: `cd packages/<package> && pnpm test -- -t "<test name>"`
- Lint: `pnpm eslint './src/**/*.{jsx,js,tsx,ts}'`

# Code Style

- TypeScript strict mode with noUnusedLocals/Parameters
- ESM modules with node-next resolution
- Use async/await with proper error handling (no floating promises)
- Objects: use shorthand notation
- Unused variables prefix: `_` (e.g., `_unused`)
- Line length: 120 characters, single quotes
- Component patterns: follow existing naming, directory structure and import patterns
- Always use proper typing - avoid `any` when possible
- Error handling: use proper error types and propagation, especially with async code
- Formatting: follows Prettier configuration