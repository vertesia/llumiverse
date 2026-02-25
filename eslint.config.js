import tseslint from "typescript-eslint";
import { defineConfig } from "eslint/config";

export default defineConfig([
  ...tseslint.configs.recommended,
  {
    rules: {
      '@typescript-eslint/ban-ts-comment': 'warn',
      '@typescript-eslint/no-empty-object-type': 'warn',
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-this-alias': 'warn',
      '@typescript-eslint/no-unused-expressions': 'warn',
      '@typescript-eslint/no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          "allowShortCircuit": true,
          "allowTernary": true
        },
      ],
      '@typescript-eslint/no-wrapper-object-types': 'warn',
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              group: [
                '../core/**',
                '../../core/**',
                '../../../core/**',
                '../../../../core/**',
              ],
              message: 'Please use "@llumiverse/core" instead of relative paths.',
            }
          ]
        }
      ],
      'prefer-const': 'warn',
    },
  },
]);
