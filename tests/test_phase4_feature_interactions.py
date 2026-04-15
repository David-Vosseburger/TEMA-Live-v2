import json
import subprocess
from pathlib import Path

import numpy as np

from tema.interactions import (
    compute_pairwise_interaction_scores,
    generate_feature_crosses,
    select_top_k_interactions,
)


def test_interaction_ranking_is_deterministic():
    x = np.array(
        [
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 0.5],
            [3.0, 4.0, 0.5],
            [4.0, 3.0, 0.5],
        ]
    )
    y = x[:, 0] * x[:, 1]
    names = ["a", "b", "c"]
    ranked_a = compute_pairwise_interaction_scores(x, y, feature_names=names)
    ranked_b = compute_pairwise_interaction_scores(x, y, feature_names=names)
    assert ranked_a == ranked_b
    assert (ranked_a[0].feature_i, ranked_a[0].feature_j) == ("a", "b")


def test_top_k_selection():
    x = np.array(
        [
            [1.0, 2.0, 3.0, 0.1],
            [2.0, 1.0, 4.0, 0.2],
            [3.0, 4.0, 1.0, 0.3],
            [4.0, 3.0, 2.0, 0.4],
            [5.0, 5.0, 5.0, 0.5],
        ]
    )
    y = x[:, 0] * x[:, 1] + 0.2 * (x[:, 1] * x[:, 2])
    scores = compute_pairwise_interaction_scores(x, y, feature_names=["a", "b", "c", "d"])
    top2 = select_top_k_interactions(scores, 2)
    assert len(top2) == 2
    assert top2[0].score >= top2[1].score


def test_feature_cross_generation_shape_and_content():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    crosses, names = generate_feature_crosses(
        x,
        selected_pairs=[("a", "b"), ("b", "c")],
        feature_names=["a", "b", "c"],
    )
    assert crosses.shape == (2, 2)
    assert names == ["a__x__b", "b__x__c"]
    assert np.allclose(crosses[:, 0], np.array([2.0, 20.0]))
    assert np.allclose(crosses[:, 1], np.array([6.0, 30.0]))


def test_phase4_interactions_entrypoint_smoke(tmp_path):
    root = Path(__file__).resolve().parent.parent
    run_id = "smoke-phase4-interactions"
    out_root = tmp_path / "outputs"
    cmd = [
        "python",
        str(root / "scripts" / "run_phase4_feature_interactions.py"),
        "--run-id",
        run_id,
        "--out-root",
        str(out_root),
        "--budget",
        "6",
        "--seed",
        "9",
        "--top-k",
        "3",
        "--no-optuna",
        "--include-crosses",
    ]
    subprocess.check_call(cmd, cwd=str(root))
    artifact = out_root / run_id / "interaction_discovery.json"
    assert artifact.exists()
    payload = json.loads(artifact.read_text())
    assert len(payload["top_interactions"]) == 3
    assert payload["cross_features"]["shape"][1] == 3
