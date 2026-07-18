"""Curated official-source audit for Route C; this module downloads no data."""
from __future__ import annotations

import pandas as pd


def build_public_candidates(accessed_on: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        {
            "candidate_id": "raise_raw_images", "dataset_name": "RAISE", "official_source": "https://loki.disi.unitn.it/RAISE/index.php",
            "official_paper": "https://iris.unitn.it/handle/11572/122245", "license": "Non-commercial research and educational use; citation required",
            "license_status": "OFFICIAL-PAGE-CONFIRMED", "image_count": 8156, "scene_structure": "4 photographers; >80 places; categories and camera metadata; no official train split",
            "format_bit_depth": "camera-native high-resolution RAW; exact bit depth camera-dependent", "download_size": "350 GB full; official 1k/2k/4k/6k subsets",
            "pmrid_isolation": "different dataset/cameras; must verify hashes and content after acquisition", "preprocessing_complexity": "high: proprietary RAW, Bayer, black level and per-camera scaling",
            "history_risk": "not found in current local project", "status": "TRAINING-SOURCE-CANDIDATE-WITH-LIMITATIONS", "limitation": "download/input audit, grouping, and RAW preprocessing still required",
        },
        {
            "candidate_id": "mit_adobe_fivek", "dataset_name": "MIT-Adobe FiveK", "official_source": "https://data.csail.mit.edu/graphics/fivek/",
            "official_paper": "https://doi.org/10.1145/2010324.1964969", "license": "Research use under two file-list-specific Adobe/MIT licenses",
            "license_status": "OFFICIAL-PAGE-CONFIRMED-FILE-SCOPED", "image_count": 5000, "scene_structure": "5,000 distinct photographs with semantic labels; no official train split",
            "format_bit_depth": "DNG RAW inputs; optional 16-bit ProPhoto RGB TIFF expert renditions", "download_size": "about 50 GB",
            "pmrid_isolation": "different dataset and photography corpus; must verify after acquisition", "preprocessing_complexity": "high: DNG/Bayer or processed wide-gamut TIFF; file-specific license mapping",
            "history_risk": "not found in current local project", "status": "TRAINING-SOURCE-CANDIDATE-WITH-LIMITATIONS", "limitation": "license file mapping, content grouping, and preprocessing must be frozen",
        },
        {
            "candidate_id": "sid_see_in_the_dark", "dataset_name": "See-in-the-Dark (SID)", "official_source": "https://github.com/cchen156/Learning-to-See-in-the-Dark",
            "official_paper": "https://arxiv.org/abs/1805.01934", "license": "MIT repository license; dataset-license scope requires final confirmation before use",
            "license_status": "SCOPE-REVIEW-REQUIRED", "image_count": "5094 short-exposure RAW + 424 long-exposure references",
            "scene_structure": "official train/validation/test prefixes; multiple short exposures per long reference", "format_bit_depth": "Sony ARW and Fuji RAF RAW; optional 16-bit processed references",
            "download_size": "Sony 25 GB; Fuji 52 GB", "pmrid_isolation": "different from PMRID, but local PMRID parent contains SID list files and historical derived patches",
            "preprocessing_complexity": "high: camera-specific RAW packing, black level, Bayer pattern and exposure ratios", "history_risk": "HIGH: historical local training tree and Sony lists",
            "status": "TRAINING-SOURCE-CANDIDATE-WITH-LIMITATIONS", "limitation": "license scope and project-history contamination require dedicated review",
        },
        {
            "candidate_id": "sidd_full", "dataset_name": "SIDD", "official_source": "https://abdokamel.github.io/sidd/",
            "official_paper": "https://www.eecs.yorku.ca/~kamel/sidd/files/SIDD_CVPR_2018.pdf", "license": "MIT License for dataset and associated repositories",
            "license_status": "OFFICIAL-PAGE-CONFIRMED", "image_count": "~30,000; 80% (~24,000) released for training",
            "scene_structure": "10 scenes; 160 released scene instances with camera/ISO/exposure/illumination naming", "format_bit_depth": "Raw-RGB MAT normalized to [0,1] after black-level subtraction; sRGB PNG also supplied",
            "download_size": "large, scene-instance archives", "pmrid_isolation": "different benchmark; post-acquisition hash/content verification required",
            "preprocessing_complexity": "medium-high: already normalized Raw-RGB MAT and camera metadata; not untouched high-bit content", "history_risk": "HIGH: SIDD sample files exist in PNGAN parent project",
            "status": "TRAINING-SOURCE-CANDIDATE-WITH-LIMITATIONS", "limitation": "only 10 base scenes, processed scale, and historical project exposure",
        },
        {
            "candidate_id": "renoir", "dataset_name": "RENOIR", "official_source": "https://ani.stat.fsu.edu/~abarbu/Renoir.html",
            "official_paper": "https://arxiv.org/abs/1409.8230", "license": "No explicit dataset license found on official dataset page",
            "license_status": "UNCONFIRMED", "image_count": "about 500 images from 120 scenes", "scene_structure": "120 scenes; about four aligned noisy/low-noise images per scene; three cameras",
            "format_bit_depth": "raw and aligned camera images; exact per-camera encoding requires audit", "download_size": "about 18.4 GB across raw/aligned archives",
            "pmrid_isolation": "different dataset/cameras; verification would still be required", "preprocessing_complexity": "medium-high: three cameras and aligned/RAW variants",
            "history_risk": "not found in current local project", "status": "EXCLUDED", "limitation": "official page gives citation instructions but no explicit data-use license",
        },
    ]
    evidence = []
    for row in rows:
        evidence.extend([
            {"candidate_id": row["candidate_id"], "evidence_type": "official_dataset_page", "url": row["official_source"], "accessed_on": accessed_on, "evidence_summary": f"Dataset structure, scale and use terms recorded for {row['dataset_name']}."},
            {"candidate_id": row["candidate_id"], "evidence_type": "official_paper_or_publisher", "url": row["official_paper"], "accessed_on": accessed_on, "evidence_summary": f"Primary publication supporting dataset description for {row['dataset_name']}."},
        ])
    return pd.DataFrame(rows), pd.DataFrame(evidence)
