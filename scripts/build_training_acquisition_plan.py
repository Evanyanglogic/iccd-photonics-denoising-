"""Create the Route B acquisition plans and metadata template."""
from __future__ import annotations

import pandas as pd


def _plan(name: str, scenes: int, frames: int, config: dict) -> pd.DataFrame:
    acquisition = config["acquisition"]
    files = scenes * frames
    height, width = acquisition["nominal_shape"]
    gib = files * height * width * acquisition["bytes_per_pixel"] / (1024**3)
    values = {
        "strategy_name": name,
        "formal_source_name": acquisition["formal_name"],
        "allowed_role": acquisition["source_role"],
        "independent_scene_count": scenes,
        "frames_per_scene": frames,
        "expected_file_count": files,
        "nominal_dtype": acquisition["nominal_dtype"],
        "nominal_shape": f"{height}x{width}",
        "estimated_uncompressed_gib": round(gib, 3),
        "scene_categories": "low/mid/high brightness;flat;texture;fine lines;edges;natural objects;industrial targets;multiple spatial frequencies;multiple dynamic ranges",
        "grouping_rule": "one scene_id per independently composed scene; all repeats remain in one acquisition_group and one isolation_block",
        "pmrid_isolation": "dataset-level isolation; no PMRID content and no ICCD evaluation composition reuse",
        "capture_rule": "high-bit-depth grayscale-capable camera; low gain where practical; stable exposure; no clipping; preserve camera-native files and metadata",
        "processing_rule": "no dark/pedestal/p99/min-max/gamma correction before a separate input audit freezes preprocessing",
        "status": "TRAINING-SOURCE-CANDIDATE-WITH-LIMITATIONS",
    }
    return pd.DataFrame([{"field": key, "value": value} for key, value in values.items()])


def build_minimum_plan(config: dict) -> pd.DataFrame:
    a = config["acquisition"]
    return _plan("minimum", a["minimum_scenes"], a["minimum_frames_per_scene"], config)


def build_recommended_plan(config: dict) -> pd.DataFrame:
    a = config["acquisition"]
    return _plan("recommended", a["recommended_scenes"], a["recommended_frames_per_scene"], config)


def build_metadata_template() -> pd.DataFrame:
    fields = [
        ("content_id", "string", True, "stable unique identifier"),
        ("scene_id", "string", True, "independently composed scene identifier"),
        ("acquisition_group", "string", True, "all repeat frames from one capture group"),
        ("isolation_block_id", "string", True, "must equal or contain scene/acquisition grouping"),
        ("allowed_role", "enum", True, "training_content_only"),
        ("device_make", "string", True, "camera manufacturer"),
        ("device_model", "string", True, "camera model"),
        ("device_serial", "string", False, "record when available"),
        ("capture_date_time", "ISO-8601", True, "capture timestamp and timezone"),
        ("exposure_time", "string", True, "camera-reported exposure"),
        ("analog_gain", "string", True, "camera-reported gain or ISO"),
        ("temperature", "string", False, "sensor/body temperature when available"),
        ("readout_mode", "string", True, "mode, bit depth, binning and ROI"),
        ("black_level_setting", "string", False, "camera setting or unknown; never infer"),
        ("lens", "string", True, "lens and aperture where applicable"),
        ("software", "string", True, "capture/export software and version"),
        ("internal_correction_status", "enum", True, "enabled/disabled/unknown for each camera correction"),
        ("original_filename", "string", True, "camera-native filename"),
        ("relative_path", "string", True, "path under acquisition root"),
        ("file_size", "integer", True, "bytes"),
        ("sha256", "hex", True, "hash of immutable source file"),
        ("dtype", "string", True, "audited after acquisition"),
        ("shape", "string", True, "audited after acquisition"),
        ("processing_status", "enum", True, "likely raw-like/likely camera-processed/unknown after audit"),
        ("scene_description", "string", True, "content category without personal identifiers"),
        ("overlap_with_iccd_test", "boolean", True, "must be false"),
        ("notes", "string", False, "capture anomalies retained, not deleted"),
    ]
    return pd.DataFrame(fields, columns=["field", "type", "required", "definition"])
