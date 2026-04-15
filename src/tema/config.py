from dataclasses import dataclass
from typing import Optional


@dataclass
class BacktestConfig:
    # Turnover / rebalance controls (Phase 2b)
    rebalance_min_threshold: float = 0.001
    cost_aware_rebalance: bool = False
    cost_aware_rebalance_multiplier: float = 1.0
    cost_aware_alpha_lookback: int = 20

    # Penalty applied during selection/optimization: Sharpe - lambda * annualized_turnover
    turnover_penalty_lambda: float = 0.0

    # ML / position scalar controls
    ml_enabled: bool = True
    ml_position_scalar_method: str = "hmm_prob"
    ml_hmm_scalar_floor: float = 0.30
    ml_hmm_scalar_ceiling: float = 1.50
    ml_position_scalar: float = 1.0
    ml_position_scalar_auto: bool = True
    ml_position_scalar_target_vol: float = 0.10
    ml_position_scalar_max: float = 50.0

    # Vol-target scaling controls
    vol_target_enabled: bool = True
    vol_target_annual: float = 0.10
    vol_target_max_leverage: float = 12.0
    vol_target_min_leverage: float = 0.25
    vol_target_reference: str = "bl"
    vol_target_apply_to_ml: bool = False

    # Dynamic ensemble controls (Phase 1)
    ensemble_enabled: bool = False
    ensemble_lookback: int = 20
    ensemble_ridge_shrink: float = 0.15
    ensemble_min_weight: float = 0.05
    ensemble_max_weight: float = 0.90
    ensemble_regime_sensitivity: float = 0.40
    online_learning_enabled: bool = False
    online_learning_learning_rate: float = 0.10
    online_learning_l2: float = 1e-4
    online_learning_seed: int = 42

    # Stress-testing controls (Phase 5)
    stress_enabled: bool = False
    stress_seed: int = 42
    stress_n_paths: int = 200
    stress_horizon: int = 20

    # Costs
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005

    # Generic
    freq: str = "D"
    
    def total_cost_rate(self) -> float:
        return float(self.fee_rate + self.slippage_rate)
