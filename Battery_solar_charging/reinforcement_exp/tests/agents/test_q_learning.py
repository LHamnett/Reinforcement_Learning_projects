"""Tests for battery_opt.agents.RL_agents.q_learning."""

import numpy as np
import pytest

from battery_opt.agents.RL_agents.q_learning import QLearningAgent, discretise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CAPACITY = 10.0


@pytest.fixture()
def agent() -> QLearningAgent:
    return QLearningAgent(
        battery_capacity_kwh=CAPACITY,
        alpha=0.5,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=10,
    )


def _info(soc: float = 0.0, tariff: float = 0.28, surplus: float = 0.0) -> dict:
    return {"battery_soc_kwh": soc, "import_tariff": tariff, "surplus_kwh": surplus}


# ---------------------------------------------------------------------------
# discretise
# ---------------------------------------------------------------------------


class TestDiscretise:
    def test_returns_tuple_of_length_3(self) -> None:
        state = discretise(_info(), CAPACITY)
        assert isinstance(state, tuple) and len(state) == 3

    def test_soc_zero_gives_bin_zero(self) -> None:
        soc_bin, _, _ = discretise(_info(soc=0.0), CAPACITY)
        assert soc_bin == 0

    def test_soc_full_gives_bin_four(self) -> None:
        soc_bin, _, _ = discretise(_info(soc=CAPACITY), CAPACITY)
        assert soc_bin == 4

    def test_soc_mid_range(self) -> None:
        # 5 kWh / 10 kWh = 50% → bin 2
        soc_bin, _, _ = discretise(_info(soc=5.0), CAPACITY)
        assert soc_bin == 2

    def test_rate_off_peak(self) -> None:
        _, rate_bin, _ = discretise(_info(tariff=0.20), CAPACITY)
        assert rate_bin == 0

    def test_rate_shoulder(self) -> None:
        _, rate_bin, _ = discretise(_info(tariff=0.28), CAPACITY)
        assert rate_bin == 1

    def test_rate_peak(self) -> None:
        _, rate_bin, _ = discretise(_info(tariff=0.45), CAPACITY)
        assert rate_bin == 2

    def test_no_solar_surplus(self) -> None:
        _, _, solar_bin = discretise(_info(surplus=0.0), CAPACITY)
        assert solar_bin == 0

    def test_solar_surplus_above_threshold(self) -> None:
        _, _, solar_bin = discretise(_info(surplus=0.1), CAPACITY)
        assert solar_bin == 1

    def test_solar_surplus_at_threshold(self) -> None:
        # exactly at threshold (0.05) counts as no surplus
        _, _, solar_bin = discretise(_info(surplus=0.05), CAPACITY)
        assert solar_bin == 0


# ---------------------------------------------------------------------------
# QLearningAgent
# ---------------------------------------------------------------------------


class TestQLearningAgentInit:
    def test_epsilon_starts_at_epsilon_start(self, agent: QLearningAgent) -> None:
        assert agent.epsilon == pytest.approx(1.0)

    def test_q_table_empty_on_init(self, agent: QLearningAgent) -> None:
        assert len(agent.q_table) == 0


class TestQLearningAgentSelectAction:
    def test_returns_valid_action(self, agent: QLearningAgent) -> None:
        state = (0, 1, 0)
        for _ in range(20):
            a = agent.select_action(state, training=True)
            assert a in (0, 1, 2)

    def test_greedy_returns_best_action(self, agent: QLearningAgent) -> None:
        state = (0, 1, 0)
        agent.q_table[state] = np.array([0.0, 5.0, 1.0])
        assert agent.select_action(state, training=False) == 1

    def test_epsilon_one_is_random(self, agent: QLearningAgent) -> None:
        """With epsilon=1.0, actions should vary over many calls."""
        agent.epsilon = 1.0
        state = (0, 0, 0)
        agent.q_table[state] = np.array([0.0, 10.0, 0.0])  # greedy would always pick 1
        np.random.seed(0)
        actions = {agent.select_action(state, training=True) for _ in range(50)}
        assert len(actions) > 1  # should see multiple actions

    def test_epsilon_zero_is_greedy(self, agent: QLearningAgent) -> None:
        agent.epsilon = 0.0
        state = (2, 2, 1)
        agent.q_table[state] = np.array([10.0, 0.0, 0.0])
        assert agent.select_action(state, training=True) == 0


class TestQLearningAgentUpdate:
    def test_update_moves_q_toward_target(self, agent: QLearningAgent) -> None:
        state, next_state = (0, 0, 0), (0, 1, 0)
        before = agent._q(state)[2].copy()
        agent.update(state, action=2, reward=-1.0, next_state=next_state)
        after = agent._q(state)[2]
        # Q should have changed
        assert after != before

    def test_update_terminal_no_bootstrap(self, agent: QLearningAgent) -> None:
        state = (1, 2, 0)
        agent.q_table[state] = np.array([0.0, 0.0, 0.0])
        agent.update_terminal(state, action=1, reward=-2.0)
        # Without bootstrap: target = reward; update = alpha * (reward - 0)
        expected = 0.5 * (-2.0)   # alpha=0.5
        assert agent._q(state)[1] == pytest.approx(expected)

    def test_update_converges_to_true_value(self, agent: QLearningAgent) -> None:
        """Repeated updates on a single transition should converge."""
        state = (0, 0, 0)
        agent.epsilon = 0.0
        for _ in range(500):
            agent.update_terminal(state, action=0, reward=-3.0)
        # Q should converge to -3.0
        assert agent._q(state)[0] == pytest.approx(-3.0, abs=0.1)


class TestQLearningAgentDecay:
    def test_decay_reduces_epsilon(self, agent: QLearningAgent) -> None:
        before = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < before

    def test_decay_clamps_at_epsilon_end(self, agent: QLearningAgent) -> None:
        for _ in range(1000):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(agent.epsilon_end)

    def test_state_from_info_delegates_to_discretise(self, agent: QLearningAgent) -> None:
        info = _info(soc=5.0, tariff=0.45, surplus=0.2)
        expected = discretise(info, CAPACITY)
        assert agent.state_from_info(info) == expected
