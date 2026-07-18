"""Audit similarity grouping without treating algorithmic clusters as scenes."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from e2_content_manifest_lib import connected_components, read_config, resolve


def run(repo: Path, cfg: dict, output: Path) -> dict:
    source = resolve(repo, cfg["input_report"])
    pairs = pd.read_csv(source / "content_similarity.csv")
    manifest = pd.read_csv(output / "e2_content_manifest_20260717.csv")
    ids = manifest["source_pair_key"].tolist()
    matrix = pd.DataFrame(1.0, index=ids, columns=ids)
    for row in pairs.itertuples(index=False):
        matrix.loc[row.source_pair_key_a, row.source_pair_key_b] = row.center_roi_correlation
        matrix.loc[row.source_pair_key_b, row.source_pair_key_a] = row.center_roi_correlation
    matrix.index.name = "source_pair_key"
    matrix.to_csv(output / "content_similarity_matrix.csv", encoding="utf-8-sig")
    gate = cfg["high_similarity"]
    high_mask = (pairs.full_image_correlation >= gate["full_min"]) | (pairs.center_roi_correlation >= gate["roi_min"]) | (pairs.center_roi_ssim >= gate["ssim_min"])
    pairs.loc[high_mask].to_csv(output / "high_similarity_pairs.csv", index=False, encoding="utf-8-sig")
    cluster_rows, sensitivity_rows, assignments = [], [], {}
    for profile in cfg["grouping_profiles"]:
        mask = pd.Series(True, index=pairs.index)
        if "full_min" in profile: mask &= pairs.full_image_correlation >= profile["full_min"]
        if "roi_min" in profile: mask &= pairs.center_roi_correlation >= profile["roi_min"]
        labels = connected_components(ids, pairs, mask)
        sizes = pd.Series(labels).value_counts().to_dict()
        assignments[profile["name"]] = labels
        for item in ids:
            cluster_rows.append({"source_pair_key": item, "profile": profile["name"], "candidate_cluster": labels[item], "cluster_size": sizes[labels[item]], "candidate_group_only": True})
        sensitivity_rows.append({"profile": profile["name"], "edge_count": int(mask.sum()), "cluster_count": len(sizes), "largest_cluster_size": max(sizes.values()), "singleton_count": sum(v == 1 for v in sizes.values()), "candidate_group_only": True})
    pd.DataFrame(cluster_rows).to_csv(output / "candidate_similarity_clusters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity_rows).to_csv(output / "similarity_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    balanced = assignments[cfg["grouping_profiles"][1]["name"]]
    reps = []
    for label in sorted(set(balanced.values())):
        members = [item for item in ids if balanced[item] == label]
        if len(members) == 1:
            medoid = members[0]
        else:
            subset = pairs[pairs.source_pair_key_a.isin(members) & pairs.source_pair_key_b.isin(members)]
            scores = {item: subset.loc[(subset.source_pair_key_a == item) | (subset.source_pair_key_b == item), "center_roi_correlation"].mean() for item in members}
            medoid = max(scores, key=scores.get)
        reps.append({"profile": cfg["grouping_profiles"][1]["name"], "candidate_cluster": label, "representative_source_pair_key": medoid, "cluster_size": len(members), "candidate_group_only": True, "allowed_use": "qualitative_preview_or_debug_only"})
    pd.DataFrame(reps).to_csv(output / "representative_content_selection.csv", index=False, encoding="utf-8-sig")
    stable = len({row["cluster_count"] for row in sensitivity_rows}) == 1
    report = (
        "# E2 Content Grouping Audit\n\n"
        f"- Pair count: {len(pairs)}\n- Candidate groups stable across thresholds: {str(stable).lower()}\n"
        "- Cluster labels are algorithmic risk-analysis labels, not scene or acquisition identifiers.\n"
        "- The 100 files remain one blocked unknown source for split decisions.\n"
    )
    (output / "content_grouping_report.md").write_text(report, encoding="utf-8")
    return {"pair_count": len(pairs), "high_similarity_pair_count": int(high_mask.sum()), "profiles": sensitivity_rows, "stable_candidate_groups": stable}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-root", required=True); args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]; cfg,_=read_config(repo,args.config); output=resolve(repo,args.output_root)
    print(run(repo,cfg,output)); return 0


if __name__ == "__main__": raise SystemExit(main())

