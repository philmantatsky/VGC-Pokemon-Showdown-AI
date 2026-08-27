"""Isolated chess-style background expansion for live exact search.

Pondering starts only after our action has been selected.  It owns a separate
Showdown process, never delays submission, and never mutates the live session.  The
next request may consume a completed result; partial or late work is safely ignored.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

from vgc_bench.src.exact_planner import ExactNode, material_value, strategic_value
from vgc_bench.src.exact_sim import ExactShowdownBridge


@dataclass(frozen=True)
class PonderConfig:
    budget_s: float = 6.0
    max_opponent_choices: int = 96
    chance_samples: int = 1
    max_roots: int = 8

    def __post_init__(self) -> None:
        if not 0.1 <= self.budget_s <= 15.0:
            raise ValueError("ponder budget must be between 0.1 and 15 seconds")
        if self.max_opponent_choices < 1:
            raise ValueError("ponder opponent width must be positive")
        if not 1 <= self.chance_samples <= 2:
            raise ValueError("ponder chance samples must be 1 or 2")
        if not 1 <= self.max_roots <= 8:
            raise ValueError("ponder roots must be between 1 and 8")


@dataclass(frozen=True)
class PonderRoot:
    node: ExactNode
    probability: float
    label: str


@dataclass(frozen=True)
class PonderOutcome:
    root_label: str
    root_choice: str
    opponent_choice: str
    predicted_node: ExactNode
    value: float
    probability: float


@dataclass(frozen=True)
class PonderResult:
    generation: int
    root_choice: str
    roots: tuple[PonderRoot, ...]
    outcomes: tuple[PonderOutcome, ...]
    reference_values: dict[str, float]
    elapsed_s: float
    roots_completed: int
    choices_simulated: int
    truncated: bool
    cancelled: bool
    error: str | None = None


def _families(choice: str) -> tuple[str, ...]:
    families = []
    for atom in choice.split(","):
        parts = atom.strip().split()
        families.append(" ".join(parts[:2]) if parts else "pass")
    return tuple(families)


def _diverse_choices(
    choices: Sequence[str],
    limit: int,
    preferred: Sequence[str] = (),
) -> list[str]:
    """Keep likely replies first, then cover remaining move/switch families."""
    choices = list(dict.fromkeys(choices))
    if len(choices) <= limit:
        return choices
    legal = set(choices)
    selected = list(
        dict.fromkeys(choice for choice in preferred if choice in legal)
    )[:limit]
    covered = [set(), set()]
    for choice in selected:
        for slot, family in enumerate(_families(choice)[:2]):
            covered[slot].add(family)
    remaining = [choice for choice in choices if choice not in selected]
    while remaining and len(selected) < limit:
        best_index = 0
        best_gain = -1
        for index, choice in enumerate(remaining):
            families = _families(choice)
            gain = sum(
                family not in covered[slot]
                for slot, family in enumerate(families[:2])
            )
            if gain > best_gain:
                best_index, best_gain = index, gain
        choice = remaining.pop(best_index)
        selected.append(choice)
        for slot, family in enumerate(_families(choice)[:2]):
            covered[slot].add(family)
        if best_gain <= 0:
            break
    if len(selected) < limit:
        selected.extend(choice for choice in remaining if choice not in selected)
    return selected[:limit]


def _seed(generation: int, label: str, opponent: str, sample: int) -> str:
    digest = hashlib.blake2b(
        f"{generation}|{label}|{opponent}|{sample}".encode(), digest_size=8
    ).digest()
    values = [int.from_bytes(digest[index : index + 2], "big") for index in range(0, 8, 2)]
    return ",".join(str(value) for value in values)


def _lower_quartile(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return -1.0
    return ordered[max(0, math.ceil(0.25 * len(ordered)) - 1)]


class BackgroundPonder:
    """One daemon worker and its immutable eventual result."""

    def __init__(
        self,
        roots: Sequence[PonderRoot],
        root_choice: str,
        *,
        generation: int,
        config: PonderConfig | None = None,
        preferred_choices: dict[str, Sequence[str]] | None = None,
    ):
        self.roots = tuple(
            sorted(roots, key=lambda item: item.probability, reverse=True)
        )
        self.root_choice = root_choice
        self.generation = int(generation)
        self.config = config or PonderConfig()
        self.preferred_choices = {
            str(label): tuple(choices)
            for label, choices in (preferred_choices or {}).items()
        }
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._result: PonderResult | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"vgc-ponder-{self.generation}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def result_if_done(self) -> PonderResult | None:
        if not self._done.is_set():
            return None
        with self._lock:
            return self._result

    def result_now(self) -> PonderResult | None:
        """Return the latest completed-root snapshot without waiting."""
        with self._lock:
            return self._result

    def _publish(self, result: PonderResult) -> None:
        with self._lock:
            self._result = result

    def _finish(self, result: PonderResult) -> None:
        self._publish(result)
        self._done.set()

    def _run(self) -> None:
        started = time.monotonic()
        deadline = started + self.config.budget_s
        outcomes: list[PonderOutcome] = []
        references: dict[str, float] = {}
        roots_completed = 0
        choices_simulated = 0
        truncated = False
        error: str | None = None
        try:
            with ExactShowdownBridge() as bridge:
                for root in self.roots[: self.config.max_roots]:
                    if self._cancel.is_set() or time.monotonic() >= deadline:
                        truncated = True
                        break
                    p1_legal = bridge.choices(root.node.state, "p1")
                    if self.root_choice not in p1_legal:
                        continue
                    opponent_choices = _diverse_choices(
                        bridge.choices(root.node.state, "p2"),
                        self.config.max_opponent_choices,
                        self.preferred_choices.get(root.label, ()),
                    )
                    if not opponent_choices:
                        continue
                    branches: list[dict[str, Any]] = []
                    metadata: list[str] = []
                    for opponent_choice in opponent_choices:
                        for sample in range(self.config.chance_samples):
                            branches.append(
                                {
                                    "p1_choice": self.root_choice,
                                    "p2_choice": opponent_choice,
                                    "rng_seed": _seed(
                                        self.generation,
                                        root.label,
                                        opponent_choice,
                                        sample,
                                    ),
                                }
                            )
                            metadata.append(opponent_choice)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        truncated = True
                        break
                    results = bridge.simulate_batch(
                        root.node.state,
                        branches,
                        timeout_s=max(0.05, remaining),
                    )
                    choices_simulated += len(results)
                    root_values: list[float] = []
                    probability = root.probability / max(1, len(results))
                    for raw, opponent_choice in zip(results, metadata):
                        if self._cancel.is_set() or time.monotonic() >= deadline:
                            truncated = True
                            break
                        child = ExactNode.from_result(raw)
                        try:
                            value = strategic_value(child, "p1")
                        except Exception:
                            value = material_value(child, "p1")
                        root_values.append(value)
                        outcomes.append(
                            PonderOutcome(
                                root_label=root.label,
                                root_choice=self.root_choice,
                                opponent_choice=opponent_choice,
                                predicted_node=child,
                                value=value,
                                probability=probability,
                            )
                        )
                    if root_values:
                        references[root.label] = _lower_quartile(root_values)
                        roots_completed += 1
                        self._publish(
                            PonderResult(
                                generation=self.generation,
                                root_choice=self.root_choice,
                                roots=self.roots,
                                outcomes=tuple(outcomes),
                                reference_values=dict(references),
                                elapsed_s=time.monotonic() - started,
                                roots_completed=roots_completed,
                                choices_simulated=choices_simulated,
                                truncated=True,
                                cancelled=self._cancel.is_set(),
                            )
                        )
                    if truncated:
                        break
        except Exception as exc:  # Result telemetry owns failures; live play never does.
            error = f"{type(exc).__name__}: {exc}"
            truncated = True
        self._finish(
            PonderResult(
                generation=self.generation,
                root_choice=self.root_choice,
                roots=self.roots,
                outcomes=tuple(outcomes),
                reference_values=references,
                elapsed_s=time.monotonic() - started,
                roots_completed=roots_completed,
                choices_simulated=choices_simulated,
                truncated=truncated,
                cancelled=self._cancel.is_set(),
                error=error,
            )
        )
