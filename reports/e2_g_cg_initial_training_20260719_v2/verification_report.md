# E2 G/CG Initial Training Preflight

Status: `TRAINING-PREFLIGHT-NO-GO`

Calibration-only fitting selected `sigma_DN = 0 + 0.0591367574832 * reference_patch_mean_DN` with LOOCV RMSE 8.063817 DN. The model is operational and has no physical gain interpretation.

The frozen sCMOS ROI means span 12757.246-16435.023 DN, entirely outside the ICCD calibration range 935.120-2510.123 DN. Consequently all 300 CG training pairs predict sigma above 300 DN. PMRID deterministic patch means span 482.507-22169.838 DN; 16/39 exceed the same gate.

No clipping, brightness mapping, clamping, synthetic generation, model training, checkpoint selection, or real ICCD evaluation was performed. G-only training was not started because the pre-registered experiment requires all three arms and would not answer the requested G-vs-CG comparison.
