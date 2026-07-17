# E2 Formal Input and Provenance Audit

- E2 status: **INVALID**
- Clean-content files: 100
- Clean-content integrity: PASS
- Leakage gate: FAIL
- Single-pair round-trip: PASS
- Historical chain: INVALID_FOR_FORMAL_USE
- Batch generation performed: no

## Decision

Candidate D is selected: audit and fixed-sample historical replay only.
Batch generation is NO-GO until source scenes can be identified and split without scene leakage.
The historical `physical-scale` label must be replaced by `legacy_unscaled_content`; it is not a calibrated physical model.
The historical `p99` label means per-image clean-content p99 normalization to 0.25.

## Checks

- PASS `required_outputs`: input_audit/input_clean_manifest.csv,input_audit/clean_content_audit.csv,input_audit/clean_content_summary.json,input_audit/duplicate_or_near_duplicate_report.csv,generation_audit/e2_parameter_to_e1_mapping.csv,generation_audit/historical_e2_output_audit.csv,round_trip_audit/round_trip_metrics.csv,leakage_audit/leakage_summary.json
- PASS `clean_content_integrity`: PASS
- PASS `full_clean_file_hashes`: count=100
- PASS `round_trip`: PASS
- FAIL `generation_numerics`: FAIL
- PASS `historical_exact_replay`: {"status": "PASS", "generation_numeric_status": "FAIL", "variant_count": 2, "all_historical_clean_exact_match": true, "all_historical_noisy_exact_match": true, "note": "This is a fixed single-pair replay, not batch synthetic generation."}
- FAIL `scene_isolated_split`: scene/source-group isolation cannot be verified because source_scene metadata is absent
- FAIL `historical_provenance`: INVALID_FOR_FORMAL_USE
- PASS `worktree_unchanged`: before='?? configs/e2_formal_input_audit_20260717.yaml\n?? scripts/audit_e2_round_trip.py\n?? scripts/audit_e2_synthetic_inputs.py\n?? scripts/check_e2_split_leakage.py\n?? scripts/run_e2_formal_input_audit.py\n' after='?? configs/e2_formal_input_audit_20260717.yaml\n?? scripts/audit_e2_round_trip.py\n?? scripts/audit_e2_synthetic_inputs.py\n?? scripts/check_e2_split_leakage.py\n?? scripts/run_e2_formal_input_audit.py\n'
- FAIL `committed_clean_code`: required for non-smoke formal status
- PASS `no_batch_generation`: only two fixed single-pair variant replays were written
