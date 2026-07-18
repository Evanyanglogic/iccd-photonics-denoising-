# Failed Diagnostic Run

This directory is retained as a failed first attempt and must not be used as a formal validation-source result.

Failure: `NON-FINITE-THUMBNAIL-DIAGNOSTICS`.

The current Pillow path used explicit `mode="F"` construction and produced non-finite thumbnail arrays. Consequently, perceptual hashes, correlations, SSIM values, and the candidate status in this directory are invalid. Source-file protection checks passed, and no D/F source data were modified. A full source-data rerun is required in a new suffixed directory.
