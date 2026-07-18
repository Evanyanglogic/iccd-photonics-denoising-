# E1 Observed-State Reliability And Separation Audit

Status: `OBSERVED-STATE-SEPARATION-VERIFIED-WITH-LIMITATIONS`

The audit used ten 200-frame folders, the frozen 512x512 center ROI, seven pre-registered temporal subsets, and five fixed ROI positions. No denoising result, PMRID data, synthetic pair, or model training was used.

The best LOOCV model was `S1` with RMSE 17.7550 DN. Signal-conditioned strength is supported with limitations. Brightness-adjusted folder-state conditioning is not supported as a formal folder condition. Folder 5 and other high-signal folders are retained and their LOO influence is reported.

Primary split `B` freezes calibration folders `[1, 4, 7, 8, 10, 13]` and evaluation folders `[2, 5, 9, 11]`. Historical Candidate A remains an all-folder historical baseline; future `G-calibration` must estimate sigma only from primary calibration folders.

`CG_READY=true`. `CGS_READINESS=NOT-YET`. The result authorizes mathematical freezing of a signal-conditioned CG only; it does not authorize training, synthetic generation, CGS, or physical-condition claims.
