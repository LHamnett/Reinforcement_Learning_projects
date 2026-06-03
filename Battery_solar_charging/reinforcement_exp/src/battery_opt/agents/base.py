"""Agent interface used by the training and evaluation loops."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RLAgent(Protocol):
    """Structural protocol that any RL agent must satisfy.

    Implementing this protocol (via duck typing — no inheritance required)
    allows the training and evaluation loops in
    :mod:`battery_opt.env.run_simulation` to remain agent-agnostic.
    """

    def state_from_info(self, info: dict) -> tuple:
        """Convert a BatteryEnv info dict to a hashable state representation."""
        ...

    def select_action(self, state: tuple, training: bool = True) -> int:
        """Return an action integer given the current state."""
        ...

    def update(
        self, state: tuple, action: int, reward: float, next_state: tuple
    ) -> None:
        """Update internal parameters from a non-terminal transition."""
        ...

    def update_terminal(
        self, state: tuple, action: int, reward: float
    ) -> None:
        """Update internal parameters from the final step of an episode."""
        ...

    def decay_epsilon(self) -> None:
        """Decay exploration rate. Called once per episode."""
        ...
