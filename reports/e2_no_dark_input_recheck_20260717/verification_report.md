# E2 No-Dark Formal Input Recheck

- Status: **VERIFIED-INPUT-WITH-LIMITATIONS**
- Preprocessing: `raw_uint16.astype(np.float32) / 65535.0`
- Dark subtraction: no
- Scalar pedestal subtraction: no
- Per-image p99 scaling: no
- Synthetic generation: no

## Processing Assessment

- Classification: processing status unknown
- Source group: source_group_unknown
- Full-image mean range: [17036.73782682419, 19692.99443411827]
- Center-ROI p99 range: [29420.140000000014, 40856.140000000014]

## Gates

- PASS `file_count`: 100
- PASS `sha256_match`: 0
- PASS `dtype_uint16`: ['uint16']
- PASS `shape_2048x2048`: ['2048x2048']
- PASS `center_roi_coordinates`: 768,768,512,512
- PASS `zero_ratio_lt_5pct`: 0.0
- PASS `saturation_ratio_lt_1pct`: 0.00048422813415527344
- PASS `negative_ratio_zero`: 0
- PASS `no_nan_inf`: True
- PASS `no_per_image_scaling`: True
- PASS `normalization_is_float32_divide_65535`: 2.9801867640344426e-08
- PASS `no_uint8_conversion`: ['uint16']
- PASS `no_silent_clipping`: True
- PASS `inter_image_mean_not_forced`: [17036.73782682419, 19692.99443411827]
- PASS `inter_image_p99_not_forced`: [39463.0, 44785.0]
- PASS `inter_image_dynamic_range_not_forced`: [32082.0, 36856.0]
- PASS `perceptual_or_exact_duplicate_fraction_lt_20pct`: 0.0
- PASS `metadata_recorded`: 100
- PASS `clean_commit`: 853add5db8c05ad6caa320275d72f119e51db4a8
