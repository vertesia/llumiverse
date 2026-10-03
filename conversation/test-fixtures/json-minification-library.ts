export default {
    purpose:
        'Exact12 library-only cases relocated from public common fixture; same entry JSON values. These helper schemas are not published OpenAPI components.',
    entries: [
        {
            name: 'valid-Configuration',
            component: 'ConversationJsonMinificationConfiguration',
            valid: true,
            value: {
                format: 'raw_json_text',
                minimum_token_reduction: 2,
                max_code_units: 1048576,
                max_depth: 128,
                max_lexical_tokens: 262144,
            },
        },
        {
            name: 'valid-Candidate',
            component: 'ConversationJsonMinificationCandidate',
            valid: true,
            value: {
                parser: 'rfc8259-lexical-v1',
                strategy: {
                    id: 'builtin.json_minification',
                    version: '1',
                    configuration_fingerprint:
                        'sha256:da37acdcb039c1c35f02ae6f830937ec9204596023ed4921414b142276dd8a2d',
                },
                source_fingerprint: 'sha256:792817d0aebc66fd1c6efc867407b08f277bc52afc007e831c21102d38096ac5',
                transforms: [
                    {
                        entry_id: 'fixture-entry',
                        source_slice: {
                            source: {
                                conversation_id: 'json-fixture',
                                revision: 3,
                            },
                            turn_id: 'fixture-turn',
                            block_id: 'fixture-block',
                            block_fingerprint:
                                'sha256:6c3b74f65c910e4f1d82cdcc8fc864567178083ba3244dbaf35915498fdcbde6',
                            selection: {
                                kind: 'whole',
                            },
                        },
                        replacement_text: '{"n":900719925474099312345,"n":-0,"s":"\\u0061"}',
                        replacement_text_fingerprint:
                            'sha256:90d655c7545f73a02ec6989c6d283f92aeea84a10adef18c96d6806996aa25a1',
                    },
                ],
                kind: 'json_minification_candidate',
            },
        },
        {
            name: 'valid-MeasurementIdentity',
            component: 'ConversationJsonMinificationMeasurementIdentity',
            valid: true,
            value: {
                tokenizer: 'fixture-tokenizer',
                tokenizer_version: '1',
                adapter: 'fixture-native-projection',
                adapter_version: '1',
                target_model: 'fixture-model',
                method: 'exact',
            },
        },
        {
            name: 'invalid-max_code_units-1048577',
            component: 'ConversationJsonMinificationConfiguration',
            valid: false,
            value: {
                format: 'raw_json_text',
                minimum_token_reduction: 2,
                max_code_units: 1048577,
                max_depth: 128,
                max_lexical_tokens: 262144,
            },
        },
        {
            name: 'invalid-max_depth-129',
            component: 'ConversationJsonMinificationConfiguration',
            valid: false,
            value: {
                format: 'raw_json_text',
                minimum_token_reduction: 2,
                max_code_units: 1048576,
                max_depth: 129,
                max_lexical_tokens: 262144,
            },
        },
        {
            name: 'invalid-max_lexical_tokens-262145',
            component: 'ConversationJsonMinificationConfiguration',
            valid: false,
            value: {
                format: 'raw_json_text',
                minimum_token_reduction: 2,
                max_code_units: 1048576,
                max_depth: 128,
                max_lexical_tokens: 262145,
            },
        },
        {
            name: 'invalid-max_depth-0',
            component: 'ConversationJsonMinificationConfiguration',
            valid: false,
            value: {
                format: 'raw_json_text',
                minimum_token_reduction: 2,
                max_code_units: 1048576,
                max_depth: 0,
                max_lexical_tokens: 262144,
            },
        },
        {
            name: 'invalid-minimum_token_reduction-0',
            component: 'ConversationJsonMinificationConfiguration',
            valid: false,
            value: {
                format: 'raw_json_text',
                minimum_token_reduction: 0,
                max_code_units: 1048576,
                max_depth: 128,
                max_lexical_tokens: 262144,
            },
        },
        {
            name: 'unknown-candidate-kind',
            component: 'ConversationJsonMinificationCandidate',
            valid: false,
            value: {
                parser: 'rfc8259-lexical-v1',
                strategy: {
                    id: 'builtin.json_minification',
                    version: '1',
                    configuration_fingerprint:
                        'sha256:da37acdcb039c1c35f02ae6f830937ec9204596023ed4921414b142276dd8a2d',
                },
                source_fingerprint: 'sha256:792817d0aebc66fd1c6efc867407b08f277bc52afc007e831c21102d38096ac5',
                transforms: [
                    {
                        entry_id: 'fixture-entry',
                        source_slice: {
                            source: {
                                conversation_id: 'json-fixture',
                                revision: 3,
                            },
                            turn_id: 'fixture-turn',
                            block_id: 'fixture-block',
                            block_fingerprint:
                                'sha256:6c3b74f65c910e4f1d82cdcc8fc864567178083ba3244dbaf35915498fdcbde6',
                            selection: {
                                kind: 'whole',
                            },
                        },
                        replacement_text: '{"n":900719925474099312345,"n":-0,"s":"\\u0061"}',
                        replacement_text_fingerprint:
                            'sha256:90d655c7545f73a02ec6989c6d283f92aeea84a10adef18c96d6806996aa25a1',
                    },
                ],
                kind: 'unknown',
            },
        },
        {
            name: 'valid-Configuration-unknown-field',
            component: 'ConversationJsonMinificationConfiguration',
            valid: false,
            value: {
                format: 'raw_json_text',
                minimum_token_reduction: 2,
                max_code_units: 1048576,
                max_depth: 128,
                max_lexical_tokens: 262144,
                unrecognized: true,
            },
        },
        {
            name: 'valid-Candidate-unknown-field',
            component: 'ConversationJsonMinificationCandidate',
            valid: false,
            value: {
                parser: 'rfc8259-lexical-v1',
                strategy: {
                    id: 'builtin.json_minification',
                    version: '1',
                    configuration_fingerprint:
                        'sha256:da37acdcb039c1c35f02ae6f830937ec9204596023ed4921414b142276dd8a2d',
                },
                source_fingerprint: 'sha256:792817d0aebc66fd1c6efc867407b08f277bc52afc007e831c21102d38096ac5',
                transforms: [
                    {
                        entry_id: 'fixture-entry',
                        source_slice: {
                            source: {
                                conversation_id: 'json-fixture',
                                revision: 3,
                            },
                            turn_id: 'fixture-turn',
                            block_id: 'fixture-block',
                            block_fingerprint:
                                'sha256:6c3b74f65c910e4f1d82cdcc8fc864567178083ba3244dbaf35915498fdcbde6',
                            selection: {
                                kind: 'whole',
                            },
                        },
                        replacement_text: '{"n":900719925474099312345,"n":-0,"s":"\\u0061"}',
                        replacement_text_fingerprint:
                            'sha256:90d655c7545f73a02ec6989c6d283f92aeea84a10adef18c96d6806996aa25a1',
                    },
                ],
                kind: 'json_minification_candidate',
                unrecognized: true,
            },
        },
        {
            name: 'valid-MeasurementIdentity-unknown-field',
            component: 'ConversationJsonMinificationMeasurementIdentity',
            valid: false,
            value: {
                tokenizer: 'fixture-tokenizer',
                tokenizer_version: '1',
                adapter: 'fixture-native-projection',
                adapter_version: '1',
                target_model: 'fixture-model',
                method: 'exact',
                unrecognized: true,
            },
        },
    ],
};
