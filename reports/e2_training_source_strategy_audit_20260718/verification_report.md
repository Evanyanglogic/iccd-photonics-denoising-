# E2 Training Source Strategy Audit

Status: `TRAINING-STRATEGY-VERIFIED-WITH-LIMITATIONS`

## Decision

The unique primary strategy is `NEW-CONTENT-ACQUISITION`. The source remains unmaterialized and the formal training-content status remains `MISSING`. A minimum acquisition contains 20 independent scenes and 3 files per scene (60 files); the recommended acquisition contains 40 scenes and 5 files per scene (200 files). Every repeated capture remains blocked by its scene and acquisition group.

## Route A

The bounded review retained 13 local data candidates from the frozen audit. None is training-ready. The strongest local lead, `E:/PMRID-Pytorch-main/Code/data`, contains 10,666 historical 8-bit RGB PNG patches in input/groundtruth directories and SID-style list files, but not a sufficiently traceable, untouched content source. Existing sCMOS, PMRID validation, ICCD evaluation/calibration data, previews, caches, sparse public samples and model outputs keep their frozen non-training roles.

## Route B

New acquisition has the lowest controllable leakage risk because role, scene, acquisition group, device settings and hashes can be recorded before any generator use. It remains conditional on manual confirmation, actual acquisition and a separate formal input audit. The data must be called `newly acquired operational training content`, not clean ground truth.

## Route C

Five official candidates were reviewed without downloading data. RAISE and MIT-Adobe FiveK are the two backups. RAISE has explicit non-commercial research/education terms and camera-native RAW subsets; FiveK has 5,000 DNG files under file-list-specific research licenses. SID and SIDD have higher project-history/preprocessing risk. RENOIR is excluded because its official page did not expose an explicit dataset license.

## Readiness

No formal synthetic generation, split construction or model training is allowed. The next task is human confirmation of the acquisition protocol, directory layout and metadata table; acquisition must not start automatically.
