import json
from pathlib import Path

from tema.config import BacktestConfig
from tema.ml import compute_position_scalars, threshold_probabilities
from tema.pipeline import run_pipeline


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_threshold_and_scalar_mapping():
    probs = [0.2, 0.5, 0.8]
    decisions = threshold_probabilities(probs, threshold=0.5)
    assert decisions == [0.0, 1.0, 1.0]

    scalars = compute_position_scalars(probs, floor=0.3, ceiling=1.5, decisions=decisions)
    assert scalars[0] == 0.0
    assert round(scalars[1], 6) == 0.9
    assert round(scalars[2], 6) == 1.26


def test_pipeline_ml_modular_path_integration(tmp_path):
    data_dir = tmp_path / "merged_d1"
    out_root = tmp_path / "outputs"
    data_dir.mkdir()
    _write_csv(
        data_dir / "a_d1_merged.csv",
        "timestamp,close_mid\n"
        "1677628800000,100\n"
        "1677715200000,101\n"
        "1677801600000,102\n"
        "1678060800000,105\n"
        "1678147200000,108\n",
    )
    _write_csv(
        data_dir / "b_d1_merged.csv",
        "timestamp,close_mid\n"
        "1677628800000,60\n"
        "1677715200000,59\n"
        "1677801600000,58\n"
        "1678060800000,57\n"
        "1678147200000,56\n",
    )

    cfg = BacktestConfig(
        modular_data_signals_enabled=True,
        data_path=str(data_dir),
        data_max_assets=2,
        data_min_rows=4,
        data_train_ratio=0.8,
        signal_fast_period=2,
        signal_slow_period=3,
        signal_method="ema",
        ml_enabled=True,
        ml_modular_path_enabled=True,
        ml_probability_threshold=0.55,
    )
    res = run_pipeline(run_id="ml-modular-test", cfg=cfg, out_root=str(out_root))
    ml_info = json.loads((Path(res["out_dir"]) / "ml_info.json").read_text(encoding="utf-8"))
    assert ml_info["modular_path_enabled"] is True
    assert len(ml_info["scalar"]) == 2
    assert all(s >= 0.0 for s in ml_info["scalar"])
