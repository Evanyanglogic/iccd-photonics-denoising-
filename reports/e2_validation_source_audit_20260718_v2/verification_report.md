# E2 Validation Content Source Audit

Status: `VALIDATION-READY-FOUND`

The bounded inventory covered 13 configured candidate directories: 1 Priority A, 3 Priority B, and 9 Priority C. Deep review was limited to one Priority A and three Priority B candidates.

The primary candidate is the official PMRID ECCV 2020 benchmark at `E:/PMRID-Pytorch-main/PMRID/PMRID`. Its 39 GT RAW files are readable uint16 3000x4000 Bayer arrays, organized by four official scene IDs with bright/dark and ISO/exposure metadata. Its formal status is `VALIDATION-READY` based on the frozen integrity, grouping, independence, numerical, and leakage gates recorded in `verification_status.json`.

This decision does not create a split, generate synthetic pairs, establish ICCD-domain equivalence, or validate model performance. The PMRID source has only four scenes and a mobile Bayer RAW domain; future preprocessing and scene blocking must be preregistered.
