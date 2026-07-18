# Failed Semantic Audit

This first role-manifest run is retained but must not be used as the formal result.

Failure: `OVERBROAD-HISTORICAL-TRAINING-FLAG`.

The role assignments, 39-row PMRID scene manifest, and source-protection checks completed, but the builder inherited a coarse prior-audit flag into `used_in_training` for every excluded source. This overstates the known history of real ICCD evaluation frames, calibration outputs, and dark frames. A new run must use explicit per-source history assignments. No source data were modified and no synthetic pairs, splits, or training outputs were created.
