"""
Text-based reinforcement learning environment for a smart customer support system.

The environment is intentionally simple and beginner-friendly:
- The observation is a customer message.
- The action is the support agent's text reply.
- A reward between 0.0 and 1.0 is returned after each reply.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional


Scenario = Dict[str, Any]


SCENARIOS: Dict[str, List[Scenario]] = {
    "easy": [
        {
            "id": "easy_refund_damaged_item",
            "customer_query": (
                "Hi, I received a broken lamp today. Can I get a refund?"
            ),
            "required": [
                ["refund", "money back"],
                ["return", "send it back"],
                ["3-5 business days", "5 business days", "business days"],
            ],
            "optional": [
                ["help", "assist"],
                ["order number", "order id"],
            ],
            "empathy": [
                ["sorry", "apologize"],
                ["understand", "inconvenience", "frustrating"],
            ],
        },
        {
            "id": "easy_delivery_delay",
            "customer_query": (
                "My order was supposed to arrive yesterday and it still says "
                "'in transit'. Where is it?"
            ),
            "required": [
                ["track", "tracking", "check the status"],
                ["delay", "late"],
                ["order number", "tracking number"],
            ],
            "optional": [
                ["help", "assist"],
                ["carrier", "courier"],
            ],
            "empathy": [
                ["sorry", "apologize"],
                ["understand", "frustrating"],
            ],
        },
        {
            "id": "easy_password_reset",
            "customer_query": (
                "I can't log in to my account because I forgot my password. "
                "What should I do?"
            ),
            "required": [
                ["reset your password", "password reset", "reset link"],
                ["email", "registered email"],
                ["log in", "sign in", "account access"],
            ],
            "optional": [
                ["help", "assist"],
                ["secure", "security"],
            ],
            "empathy": [
                ["sorry", "apologize"],
                ["understand", "frustrating"],
            ],
        },
    ],
    "medium": [
        {
            "id": "medium_double_charge_and_address",
            "customer_query": (
                "I was charged twice for my last order, and I also need to "
                "update the shipping address for my replacement."
            ),
            "required": [
                ["charged twice", "double charge", "duplicate charge"],
                ["refund", "reverse", "billing team"],
                ["shipping address", "change the address", "update the address"],
                ["order number", "order id"],
            ],
            "optional": [
                ["replacement"],
                ["confirm", "confirmation"],
            ],
            "empathy": [
                ["sorry", "apologize"],
                ["understand", "frustrating", "concern"],
            ],
        },
        {
            "id": "medium_cancel_and_invoice",
            "customer_query": (
                "Please cancel my subscription before it renews tomorrow, and "
                "email me the final invoice."
            ),
            "required": [
                ["cancel your subscription", "cancel the subscription", "cancellation"],
                ["renewal", "renews tomorrow", "before it renews"],
                ["invoice", "final bill"],
                ["email"],
            ],
            "optional": [
                ["confirm", "confirmation"],
                ["help", "assist"],
            ],
            "empathy": [
                ["sorry", "apologize"],
                ["happy to help", "understand"],
            ],
        },
        {
            "id": "medium_missing_items_and_size",
            "customer_query": (
                "My package arrived, but two items are missing and one item I "
                "did get is the wrong size."
            ),
            "required": [
                ["missing items", "items are missing"],
                ["wrong size", "incorrect size"],
                ["replacement", "resend"],
                ["refund", "partial refund"],
            ],
            "optional": [
                ["order number", "order id"],
                ["photo", "picture"],
            ],
            "empathy": [
                ["sorry", "apologize"],
                ["understand", "inconvenience", "frustrating"],
            ],
        },
    ],
    "hard": [
        {
            "id": "hard_manager_and_compensation",
            "customer_query": (
                "This is the third time I've contacted support about the same "
                "broken headset. I'm tired of repeating myself. I want a manager "
                "and compensation."
            ),
            "required": [
                ["manager", "supervisor", "escalate"],
                ["compensation", "refund", "credit"],
                ["case", "ticket", "reference"],
                ["priority", "urgent"],
            ],
            "optional": [
                ["third time", "repeat issue", "history"],
                ["replacement"],
            ],
            "empathy": [
                ["sorry", "apologize"],
                ["understand", "frustrating", "upsetting", "disappointing"],
            ],
        },
        {
            "id": "hard_charge_and_cart_loss",
            "customer_query": (
                "Your software crashed during checkout, charged my card, and "
                "wiped my saved cart right before a client deadline. Fix this now."
            ),
            "required": [
                ["charged my card", "payment", "charge"],
                ["saved cart", "cart"],
                ["investigate", "technical team", "engineering"],
                ["urgent", "priority"],
                ["refund", "reverse", "billing review"],
            ],
            "optional": [
                ["deadline", "client deadline"],
                ["restore", "recover"],
            ],
            "empathy": [
                ["sorry", "apologize"],
                ["understand", "stressful", "frustrating"],
            ],
        },
        {
            "id": "hard_public_complaint",
            "customer_query": (
                "If nobody resolves my issue today, I'm posting the chat "
                "transcript online. My delivery never arrived, and your last "
                "agent closed the ticket without helping."
            ),
            "required": [
                ["reopen", "ticket"],
                ["delivery", "tracking", "investigate"],
                ["escalate", "supervisor", "manager"],
                ["today", "urgent", "priority"],
            ],
            "optional": [
                ["previous agent", "last agent"],
                ["compensation", "refund", "credit"],
            ],
            "empathy": [
                ["sorry", "apologize"],
                ["understand", "frustrating", "upsetting"],
            ],
        },
        {
            "id": "hard_double_charge_no_response",
            "customer_query": (
                "I was charged twice for the same order and nobody from support "
                "has replied for four days. I need a refund, an escalation, and "
                "a clear timeline for when this will be fixed."
            ),
            "required": [
                ["refund", "money back"],
                ["escalate", "supervisor", "manager"],
                ["timeline", "when this will be fixed", "within 24 hours"],
            ],
            "optional": [
                ["order id", "order number"],
            ],
            "empathy": [
                ["sorry", "apologize"],
                ["understand", "frustrating", "upsetting"],
            ],
        },
    ],
}


def normalize_text(text: str) -> str:
    """Lower-case and simplify text for lightweight keyword matching."""

    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


class SmartCustomerSupportEnv:
    """A small OpenEnv-style text environment for customer support training."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.random = random.Random(seed)
        self.current_level: Optional[str] = None
        self.current_scenario: Optional[Scenario] = None
        self.last_action: Optional[str] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.done: str = "false"

    def reset(self, level: str, index: Optional[int] = None) -> str:
        """
        Start a new episode and return the initial customer query.

        Args:
            level: One of "easy", "medium", or "hard".
            index: Optional deterministic scenario index for non-repeating runs.
        """

        if level not in SCENARIOS:
            valid_levels = ", ".join(sorted(SCENARIOS))
            raise ValueError(f"Unknown level '{level}'. Choose from: {valid_levels}")

        scenarios = SCENARIOS[level]
        self.current_level = level
        if index is None:
            self.current_scenario = self.random.choice(scenarios)
        else:
            self.current_scenario = scenarios[index % len(scenarios)]
        self.last_action = None
        self.last_result = None
        self.done = "false"
        return self.current_scenario["customer_query"]

    def step(self, action: str) -> Dict[str, Any]:
        """
        Evaluate the agent reply and end the episode.

        Returns:
            {
                "reward": float in [0.0, 1.0],
                "done": bool,
                "info": dict
            }
        """

        if self.current_scenario is None or self.current_level is None:
            raise RuntimeError("Call reset(level) before step(action).")

        if self.done == "true":
            raise RuntimeError("Episode already finished. Call reset(level) first.")

        self.last_action = action.strip()
        evaluation = self._evaluate_response(self.last_action)
        reward = float(evaluation["reward"])

        reward = max(0.02, min(reward, 0.94))

        self.last_result = {
            "reward": reward,
            "info": evaluation["info"],
        }
        self.done = "true"

        return {
            "reward": reward,
            "done": "true",
            "info": evaluation["info"],
        }

    def state(self) -> Dict[str, Any]:
        """Return the current environment state for debugging or logging."""

        history: List[Dict[str, str]] = []

        if self.current_scenario is not None:
            history.append(
                {"role": "customer", "text": self.current_scenario["customer_query"]}
            )

        if self.last_action:
            history.append({"role": "agent", "text": self.last_action})

        return {
            "level": self.current_level,
            "scenario_id": None if self.current_scenario is None else self.current_scenario["id"],
            "customer_query": (
                None
                if self.current_scenario is None
                else self.current_scenario["customer_query"]
            ),
            "last_action": self.last_action,
            "last_result": self.last_result,
            "done": self.done,
            "history": history,
        }

    def _evaluate_response(self, action: str) -> Dict[str, Any]:
        """Score the reply with weighted keyword coverage and simple penalties."""

        assert self.current_scenario is not None
        assert self.current_level is not None

        normalized_action = normalize_text(action)
        word_count = len(normalized_action.split())
        weak_inputs = {"ok", "idk", "k", "hi", "hello", "test", "n a", "na"}

        if not normalized_action or word_count < 3 or normalized_action in weak_inputs:
            reward = 0.03
            reward = max(0.02, min(reward, 0.94))
            info = {
                "message": "Weak or invalid response.",
                "expected_keywords": self._expected_keywords(),
                "matched_required": [],
                "missing_required": self._flatten_groups(self.current_scenario["required"]),
                "matched_optional": [],
                "matched_empathy": [],
                "word_count": str(word_count),
            }
            return {"reward": reward, "info": info}

        required_result = self._score_groups(
            normalized_action, self.current_scenario["required"]
        )
        optional_result = self._score_groups(
            normalized_action, self.current_scenario["optional"]
        )
        empathy_result = self._score_groups(
            normalized_action, self.current_scenario["empathy"]
        )

        length_score = 0.2 if 15 <= word_count <= 80 else 0.05
        empathy_score = 0.2 if empathy_result["matched"] else 0.05

        reward = (
            required_result["coverage"] * 0.5
            + optional_result["coverage"] * 0.1
            + empathy_score
            + length_score
        )

        if reward >= 0.95:
            reward = 0.94
        elif reward <= 0.0:
            reward = 0.02

        reward = max(0.02, min(reward, 0.94))

        if word_count < 5:
            reward = max(reward - 0.2, 0.03)

        if re.search(r"\bidk\b", normalized_action):
            reward = max(reward - 0.3, 0.03)

        if reward >= 0.95:
            reward = 0.94
        elif reward <= 0.0:
            reward = 0.02

        reward = max(0.02, min(reward, 0.94))

        info = {
            "expected_keywords": self._expected_keywords(),
            "matched_required": required_result["matched"],
            "missing_required": required_result["missing"],
            "matched_optional": optional_result["matched"],
            "matched_empathy": empathy_result["matched"],
            "word_count": str(word_count),
        }

        return {"reward": reward, "info": info}

    def _expected_keywords(self) -> Dict[str, List[str]]:
        """Return a flattened view of the scenario keywords for debugging."""

        assert self.current_scenario is not None
        return {
            "required": self._flatten_groups(self.current_scenario["required"]),
            "optional": self._flatten_groups(self.current_scenario["optional"]),
            "empathy": self._flatten_groups(self.current_scenario["empathy"]),
        }

    @staticmethod
    def _flatten_groups(groups: List[List[str]]) -> List[str]:
        return [" / ".join(group) for group in groups]

    def _score_groups(
        self, normalized_action: str, groups: List[List[str]]
    ) -> Dict[str, Any]:
        """
        A concept group is counted as matched if any phrase in the group appears.

        Example:
            ["refund", "money back"] counts as one concept.
        """

        matched = []
        missing = []

        for group in groups:
            matched_phrase = self._match_group(normalized_action, group)
            readable_group = " / ".join(group)

            if matched_phrase is not None:
                matched.append(readable_group)
            else:
                missing.append(readable_group)

        total_groups = len(groups)

        if total_groups == 0:
            coverage = 0.5
        else:
            matched_count = len(matched)

            if matched_count == total_groups:
                coverage = 0.94
            elif matched_count == 0:
                coverage = 0.02
            else:
                coverage = matched_count / total_groups

        coverage = max(0.02, min(coverage, 0.94))

        return {
            "matched": matched,
            "missing": missing,
            "coverage": coverage,
        }

    def _match_group(self, normalized_action: str, phrases: List[str]) -> Optional[str]:
        """Return the first matching phrase from a concept group, if any."""

        for phrase in phrases:
            normalized_phrase = normalize_text(phrase)
            pattern = r"\b" + re.escape(normalized_phrase).replace(r"\ ", r"\s+") + r"\b"
            if re.search(pattern, normalized_action):
                return phrase

        return None


if __name__ == "__main__":
    env = SmartCustomerSupportEnv(seed=7)
    observation = env.reset("easy")
    print("Observation:", observation)

    reply = (
        "I'm sorry about the damaged lamp. I can help with a return and refund. "
        "Please share your order number and we will process it within 3-5 business days."
    )
    result = env.step(reply)

    print("Result:", result)
    print("State:", env.state())
