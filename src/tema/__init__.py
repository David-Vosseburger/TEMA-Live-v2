# tema package init
from .config import BacktestConfig
from .turnover import should_rebalance, apply_rebalance_gating
from .runner import Runner
from .optimization import run_bayesian_optimization, run_and_write_optimization, compute_objective
from .ensemble import DynamicEnsembleConfig, compute_dynamic_ensemble_weights, combine_stream_signals
from .interactions import InteractionScore, compute_pairwise_interaction_scores, select_top_k_interactions, generate_feature_crosses
from .online_learning import OnlineLearningConfig, OnlineLogisticLearner
from .stress import compute_scenario_metrics, evaluate_stress_scenarios, historical_shock_scenarios, sample_scenario_paths

__all__ = [
    "BacktestConfig",
    "should_rebalance",
    "apply_rebalance_gating",
    "Runner",
    "run_bayesian_optimization",
    "run_and_write_optimization",
    "compute_objective",
    "DynamicEnsembleConfig",
    "compute_dynamic_ensemble_weights",
    "combine_stream_signals",
    "InteractionScore",
    "compute_pairwise_interaction_scores",
    "select_top_k_interactions",
    "generate_feature_crosses",
    "OnlineLearningConfig",
    "OnlineLogisticLearner",
    "compute_scenario_metrics",
    "historical_shock_scenarios",
    "sample_scenario_paths",
    "evaluate_stress_scenarios",
]
