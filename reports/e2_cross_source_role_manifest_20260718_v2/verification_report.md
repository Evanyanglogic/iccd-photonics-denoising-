# E2 Cross-Source Role Manifest

Status: `ROLE-MANIFEST-VERIFIED-WITH-LIMITATIONS`

The manifest freezes 15 audited sources or risk-reference entries. The current 100-image sCMOS source remains `debug_only`; all 39 PMRID benchmark GT RAW entries remain `validation_content_only` in four official scene blocks; and the formal training-content placeholder remains `MISSING`. No source is assigned `training_content_only`.

PMRID preprocessing remains `NOT-FROZEN`, and PMRID content is mobile Bayer RAW rather than ICCD-domain ground truth. Therefore formal split construction, synthetic generation, model training, checkpoint selection, and real ICCD evaluation remain not allowed.
