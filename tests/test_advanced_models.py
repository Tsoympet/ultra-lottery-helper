import numpy as np
import pandas as pd

from src.advanced_models import (
    ReinforcementLearner,
    MultiArmedBandit,
    ab_test_summary,
    backtest_strategy,
    confidence_scores,
    cross_validate_sequences,
    feature_importance_from_frequency,
    genetic_optimize,
    lstm_sequence_predict,
    model_performance_visualization,
    neural_ensemble,
    statistical_significance_test,
    transformer_sequence_predict,
)


def test_lstm_and_transformer_predictions_shape():
    history = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    lstm_preds = lstm_sequence_predict(history, window=2, steps=2)
    trans_preds = transformer_sequence_predict(history, steps=2)
    assert len(lstm_preds) == 2
    assert len(trans_preds) == 2
    assert len(lstm_preds[0]) == len(history[0])
    assert len(trans_preds[0]) == len(history[0])


def test_neural_ensemble_weighting():
    preds = [[1.0, 1.0], [3.0, 3.0]]
    combined = neural_ensemble(preds, weights=[0.25, 0.75])
    assert np.allclose(combined, [2.5, 2.5])


def test_reinforcement_and_bandit_updates():
    np.random.seed(0)
    rl = ReinforcementLearner(n_actions=2, epsilon=0.0)
    action = rl.select_action()
    updated = rl.update(action, reward=1.0)
    assert action in (0, 1)
    assert updated > 0

    bandit = MultiArmedBandit(n_arms=2, epsilon=0.0)
    arm = bandit.select_arm()
    value = bandit.update(arm, reward=0.5)
    assert bandit.counts[arm] == 1
    assert value == bandit.values[arm]


def test_genetic_optimize_prefers_higher_fitness():
    population = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    def fitness_fn(x):
        return float(np.sum(x))

    best = genetic_optimize(population, fitness_fn=fitness_fn, generations=3, seed=42)
    assert len(best) == 2
    assert fitness_fn(best) >= 3.5


def test_cross_validation_and_backtesting():
    dataset = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
    cv = cross_validate_sequences(dataset, lambda d: lstm_sequence_predict(d, steps=1), folds=2)
    assert cv["folds"] > 0
    assert cv["mae"] is not None

    def predictor(seq):
        return transformer_sequence_predict(seq, steps=1)

    bt = backtest_strategy(dataset, predictor_fn=predictor)
    assert 0.0 <= bt["avg_hit_rate"] <= 1.0
    assert bt["trials"] > 0


def test_significance_confidence_explainability_and_ab():
    sig = statistical_significance_test((0.1, 100), (0.2, 120))
    assert 0.0 <= sig["p_value"] <= 1.0

    conf = confidence_scores([1.0, 2.0, 3.0])[0]
    assert np.isclose(sum(conf), 1.0)

    feats = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0]})
    importance = feature_importance_from_frequency(feats, outcomes=[1, 0, 1])
    assert set(importance.keys()) == {"a", "b"}

    viz = model_performance_visualization([0.1, 0.2, 0.3])
    assert viz["x"] == [0, 1, 2]
    assert viz["y"] == [0.1, 0.2, 0.3]

    ab = ab_test_summary([0.1, 0.2], [0.15, 0.25])
    assert "p_value" in ab
    assert ab["variant_mean"] > ab["control_mean"]
