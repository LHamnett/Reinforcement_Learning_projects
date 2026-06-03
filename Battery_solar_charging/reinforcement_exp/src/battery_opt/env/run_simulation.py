"""Agent-agnostic training and evaluation loops for BatteryEnv."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from battery_opt.agents.base import RLAgent
from battery_opt.agents.RL_agents.q_learning import QLearningAgent
from battery_opt.env.battery_env import BatteryEnv
from battery_opt.env.create_episodes import make_episodes


def train_agent(
    train_env: BatteryEnv,
    agent: RLAgent,
    n_epochs: int = 10,
    max_episodes: int | None = None,
) -> list[float]:
    """Train ``agent`` on the training episodes for ``n_epochs``.

    Each epoch samples ``max_episodes`` customers without replacement
    (all customers if ``max_episodes`` is ``None``), shuffled each epoch
    so no episode is repeated within a single epoch.
    :meth:`~RLAgent.decay_epsilon` is called once after each episode.

    Args:
        train_env: BatteryEnv loaded with training episodes.
        agent: Any object satisfying the :class:`~battery_opt.agents.RLAgent`
            protocol.
        n_epochs: Number of passes through the (sampled) training customers.
        max_episodes: Maximum episodes to run per epoch. Pass a small number
            (e.g. ``10``) for fast debugging. ``None`` uses all customers.

    Returns:
        List of total cost per epoch (summed across sampled customers).
    """
    all_ids = np.array(list(train_env.episodes.keys()))
    n_eps   = min(max_episodes, len(all_ids)) if max_episodes else len(all_ids)
    rng     = np.random.default_rng()
    epoch_costs: list[float] = []

    train_env.set_skip_obs(True)  # tabular agents don't use the obs
    epoch_bar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        rng.shuffle(all_ids)
        episode_ids = all_ids[:n_eps]   # first n_eps after shuffle = without replacement
        epoch_cost  = 0.0

        ep_bar = tqdm(episode_ids, desc=f"  Epoch {epoch + 1}/{n_epochs}",
                      unit="ep", leave=False)
        for cid in ep_bar:
            obs, info = train_env.reset(options={"customer_id": int(cid)})
            state = agent.state_from_info(info)

            while True:
                action = agent.select_action(state, training=True)
                obs, reward, terminated, truncated, info = train_env.step(action)
                epoch_cost -= reward  # reward = -cost

                if truncated or terminated:
                    agent.update_terminal(state, action, reward)
                    break

                next_state = agent.state_from_info(info)
                agent.update(state, action, reward, next_state)
                state = next_state

            agent.decay_epsilon()
            ep_bar.set_postfix(cost=f"A${epoch_cost / (ep_bar.n or 1):.2f}")

        epoch_costs.append(epoch_cost)
        epoch_bar.set_postfix(avg_cost=f"A${epoch_cost / n_eps:.2f}")

    train_env.set_skip_obs(False)  # restore for evaluation / neural agents
    return epoch_costs


def evaluate_agent(
    val_env: BatteryEnv,
    agent: RLAgent,
    max_episodes: int | None = None,
) -> dict[int, dict]:
    """Evaluate ``agent`` greedily on episodes in ``val_env``.

    Args:
        val_env: BatteryEnv loaded with validation episodes.
        agent: Any object satisfying the :class:`~battery_opt.agents.RLAgent`
            protocol.
        max_episodes: Maximum customers to evaluate. ``None`` evaluates all.

    Returns:
        Dict mapping customer_id → metrics dict with keys
        ``total_cost``, ``baseline_cost``, ``saving``, ``saving_pct``.
    """
    items = list(val_env.episodes.items())
    if max_episodes:
        items = items[:max_episodes]

    results: dict[int, dict] = {}

    for cid, episode_df in tqdm(items, desc="Evaluating", unit="ep"):
        obs, info = val_env.reset(options={"customer_id": cid})
        state = agent.state_from_info(info)
        total_cost = 0.0

        while True:
            action = agent.select_action(state, training=False)
            obs, reward, terminated, truncated, info = val_env.step(action)
            total_cost -= reward

            if truncated or terminated:
                break

            state = agent.state_from_info(info)

        baseline_cost = episode_df["import_cost_solar_used_immediate"].sum()
        saving = baseline_cost - total_cost
        results[cid] = {
            "total_cost":    total_cost,
            "baseline_cost": baseline_cost,
            "saving":        saving,
            "saving_pct":    100 * saving / baseline_cost if baseline_cost > 0 else 0.0,
        }

    return results


def print_eval_summary(results: dict[int, dict]) -> None:
    """Print aggregate evaluation metrics across all customers."""
    total_cost     = sum(r["total_cost"]    for r in results.values())
    total_baseline = sum(r["baseline_cost"] for r in results.values())
    total_saving   = total_baseline - total_cost
    n              = len(results)

    avg_saving        = total_saving / n
    avg_saving_pct    = sum(r["saving_pct"] for r in results.values()) / n

    print(f"{'=' * 52}")
    print(f"Validation results  ({n} customers)")
    print(f"  Avg cost/customer  : A${total_cost / n:.2f}")
    print(f"  Avg saving/customer: A${avg_saving:.2f} ({avg_saving_pct:.1f}%)")
    print(f"  Overall saving     : A${total_saving:.2f} "
          f"({100 * total_saving / total_baseline:.1f}%)")
    print(f"{'=' * 52}")


def plot_agent_analysis(
    val_env: BatteryEnv,
    agent: RLAgent,
    n_samples: int = 5,
    hours: int = 744,
    seed: int | None = None,
) -> None:
    """Plot energy, battery and cost analysis for a random sample of validation episodes.

    For each sampled customer, four panels are drawn (sharing the x-axis):

    1. **Energy** — hourly consumption vs solar generation
    2. **Battery SoC** — state of charge over time
    3. **Actions** — discharge / idle / charge decision at each hour
    4. **Cumulative cost** — Q-learning agent vs solar-only baseline

    Args:
        val_env: BatteryEnv loaded with validation episodes.
        agent: Trained agent to evaluate greedily.
        n_samples: Number of randomly sampled customers to plot.
        hours: Number of hours to display per plot (default 744 = one month).
        seed: Random seed for reproducible customer sampling.
    """
    rng = np.random.default_rng(seed)
    all_cids   = list(val_env.episodes.keys())
    sample_cids = rng.choice(all_cids, size=min(n_samples, len(all_cids)), replace=False)

    action_labels  = {0: "Discharge", 1: "Idle", 2: "Charge"}
    action_colours = {0: "#e74c3c",   1: "#bdc3c7", 2: "#27ae60"}

    for cid in sample_cids:
        cid = int(cid)
        h = min(hours, val_env._ep_n_steps[cid])

        consumption = val_env._ep_consumption[cid][:h]
        solar       = val_env._ep_solar[cid][:h]
        tariff      = val_env._ep_tariff[cid][:h]

        # Greedy rollout — collect per-step traces
        obs, info = val_env.reset(options={"customer_id": cid})
        state = agent.state_from_info(info)
        soc_trace, actions, agent_costs = [], [], []

        for _ in range(h):
            action = agent.select_action(state, training=False)
            obs, reward, terminated, truncated, info = val_env.step(action)
            soc_trace.append(info["battery_soc_kwh"])
            actions.append(action)
            agent_costs.append(-reward)
            if truncated or terminated:
                break
            state = agent.state_from_info(info)

        n = len(soc_trace)
        x = np.arange(n)
        baseline_costs = np.maximum(consumption[:n] - solar[:n], 0) * tariff[:n]

        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        fig.suptitle(
            f"Customer {cid}  |  first {n} hours  "
            f"|  saving vs baseline: "
            f"A${(baseline_costs.sum() - sum(agent_costs)):.2f}",
            fontsize=11,
        )

        # 1. Energy
        axes[0].plot(x, consumption[:n], label="Consumption", color="steelblue", alpha=0.8, lw=0.8)
        axes[0].plot(x, solar[:n],       label="Solar (GG)",  color="orange",    alpha=0.8, lw=0.8)
        axes[0].set_ylabel("kWh")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # 2. Battery SoC
        axes[1].plot(x, soc_trace, color="purple", lw=0.9)
        axes[1].axhline(val_env.battery_capacity_kwh, color="red",
                        linestyle="--", alpha=0.5, label="Capacity")
        axes[1].set_ylabel("SoC (kWh)")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # 3. Actions
        for a in [2, 1, 0]:   # charge on top, discharge on bottom
            mask = np.array(actions) == a
            axes[2].scatter(x[mask], np.full(mask.sum(), a),
                            c=action_colours[a], s=6, marker="|",
                            label=action_labels[a])
        axes[2].set_yticks([0, 1, 2])
        axes[2].set_yticklabels(["Discharge", "Idle", "Charge"])
        axes[2].legend(loc="upper right", fontsize=8, markerscale=3)
        axes[2].grid(True, alpha=0.3)

        # 4. Cumulative cost
        axes[3].plot(x, np.cumsum(agent_costs),    label="Q-learning",          color="steelblue", lw=1)
        axes[3].plot(x, np.cumsum(baseline_costs), label="Baseline (solar only)", color="orange", linestyle="--", lw=1)
        axes[3].set_ylabel("Cumulative cost (A$)")
        axes[3].set_xlabel("Hour")
        axes[3].legend(loc="upper left", fontsize=8)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import pandas as pd
    from battery_opt.data.create_episodes import load_config, make_episodes
    from battery_opt.env.battery_env import ObsNormaliser
    from pathlib import Path

    config = load_config(Path(__file__).resolve().parents[3] / "configs" / "default.yaml")
    project_root = Path(__file__).resolve().parents[3]  # …/reinforcement_exp/
    combined_dir = project_root / config["data"]["combined_dir"]
    costs_df = pd.read_csv(combined_dir / "augrid_costs.csv", parse_dates=["timestamp"])

    train_episodes, val_episodes, _ = make_episodes(
        costs_df,
        train_ids=config["data_splits"]["train_ids"],
        val_ids=config["data_splits"]["val_ids"],
        test_ids=config["data_splits"]["test_ids"],
    )

    normaliser = ObsNormaliser.from_episodes(train_episodes)
    bat = config["battery"]
    train_env = BatteryEnv(train_episodes, normaliser,
                           battery_capacity_kwh=bat["capacity_kwh"],
                           charge_efficiency=bat["charge_efficiency"],
                           discharge_efficiency=bat["discharge_efficiency"],
                           max_charge_rate_kw=bat["max_charge_rate_kw"],
                           max_discharge_rate_kw=bat["max_discharge_rate_kw"])

    val_env = BatteryEnv(val_episodes, normaliser,
                         battery_capacity_kwh=bat["capacity_kwh"],
                         charge_efficiency=bat["charge_efficiency"],
                         discharge_efficiency=bat["discharge_efficiency"],
                         max_charge_rate_kw=bat["max_charge_rate_kw"],
                         max_discharge_rate_kw=bat["max_discharge_rate_kw"])

    agent = QLearningAgent(battery_capacity_kwh=bat["capacity_kwh"])
    # train_agent(train_env, agent, n_epochs=1, max_episodes=10)
    # results = evaluate_agent(val_env, agent, max_episodes=10)

    train_agent(train_env, agent, n_epochs=1, max_episodes=None)
    results = evaluate_agent(val_env, agent, max_episodes=None)

    print_eval_summary(results)
    plot_agent_analysis(val_env, agent, n_samples=5, hours=24*365, seed=42)
