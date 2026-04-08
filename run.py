"""
Runner for the smart customer support environment.

This script supports two modes:
- baseline mode: prints average rewards for evaluation
- demo mode: prints full interactions for debugging and presentation
"""

from __future__ import annotations

import random
import sys
from typing import Any, Callable, Dict

from env import SmartCustomerSupportEnv


# Change this to "baseline" when you want the compact evaluation summary.
MODE = "demo"  # "demo" or "baseline"

# Used in baseline mode for average-reward evaluation.
BASELINE_EPISODES = 20

# Used in demo mode for readable step-by-step examples.
DEMO_EPISODES = 3
DEMO_POLICY_NAME = "rule_based"


RANDOM_RESPONSES = [
    "We are looking into it.",
    "Please wait while I check.",
    "Sorry for the issue. Share your order number.",
    "I can help with that.",
    "We will update you soon.",
]


def clamp_score(score: float) -> float:
    """Keep every reported score strictly inside the allowed grading range."""

    return max(0.01, min(float(score), 0.99))


def random_agent(_: str) -> str:
    """Return a random generic response."""

    return random.choice(RANDOM_RESPONSES)


def configure_output_encoding() -> None:
    """Allow Unicode labels to print cleanly on common terminals."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def rule_based_agent(observation: str) -> str:
    """A tiny baseline policy with hand-written support replies."""

    text = observation.lower()

    if "broken lamp" in text or "refund" in text:
        return (
            "I'm sorry your lamp arrived damaged. I can help with a return and "
            "refund. Please send your order number, and the refund will be "
            "processed within 3-5 business days."
        )

    if "in transit" in text or "arrive yesterday" in text:
        return (
            "I'm sorry about the delay. I can check the tracking status for you. "
            "Please share your order number or tracking number so I can review "
            "the late delivery."
        )

    if "forgot my password" in text or "can't log in" in text:
        return (
            "I'm sorry you're having trouble signing in. Please use the password "
            "reset link with your registered email to reset your password and "
            "regain account access."
        )

    if "charged twice" in text and "shipping address" in text:
        return (
            "I'm sorry about the duplicate charge and the address issue. Please "
            "share your order number so I can help reverse the extra charge and "
            "update the shipping address for your replacement. I will send a "
            "confirmation once both steps are handled."
        )

    if "cancel my subscription" in text:
        return (
            "I can help cancel the subscription before renewal tomorrow and email "
            "your final invoice. Please confirm the account email so I can process "
            "the cancellation and send confirmation."
        )

    if "missing" in text and "wrong size" in text:
        return (
            "I'm sorry for the inconvenience. Please share your order number and, "
            "if possible, a photo. I can help with a replacement for the wrong "
            "size item, resend the missing items, and arrange a partial refund if needed."
        )

    if "manager and compensation" in text or "third time" in text:
        return (
            "I'm very sorry you've had to repeat this issue. I understand how "
            "frustrating this is. I will escalate this case to a manager as a "
            "priority, keep the ticket active, and review compensation options "
            "such as a refund or account credit."
        )

    if "charged my card" in text and "saved cart" in text:
        return (
            "I'm sorry about the charge and the lost cart, especially with your "
            "deadline. I will treat this as urgent, ask our technical team to "
            "investigate the cart issue, and start a billing review to reverse "
            "or refund the payment."
        )

    if "posting the chat transcript online" in text or "delivery never arrived" in text:
        return (
            "I'm sorry this has been so upsetting. I will reopen the ticket, "
            "treat this as urgent today, investigate the delivery tracking, and "
            "escalate the case to a supervisor so we can resolve it properly."
        )

    return (
        "I'm sorry for the inconvenience. Please share your order number or "
        "account email and I will assist you."
    )


def run_policy(
    env: SmartCustomerSupportEnv,
    level: str,
    policy: Callable[[str], str],
    episodes: int,
) -> float:
    """Run multiple episodes and return the average reward."""

    if episodes <= 0:
        return 0.01

    total_reward = 0.0

    for _ in range(episodes):
        observation = env.reset(level)
        action = policy(observation)
        result = env.step(action)
        total_reward += clamp_score(result["reward"])

    average_reward = total_reward / episodes
    average_reward = clamp_score(average_reward)
    return average_reward


def print_list_items(items: list[str], bullet: str = "*") -> None:
    """Print one item per line, or 'None' when the list is empty."""

    if not items:
        print("  None")
        return

    for item in items:
        print(f"  {bullet} {item}")


def print_expected_keywords(expected_keywords: Dict[str, list[str]]) -> None:
    """Print expected required, optional, and empathy keyword groups."""

    print("\U0001f3af Expected:")
    print("Required:")
    print_list_items(expected_keywords["required"], bullet="-")
    print("Optional:")
    print_list_items(expected_keywords["optional"], bullet="-")
    print("Empathy:")
    print_list_items(expected_keywords["empathy"], bullet="-")


def print_detailed_info(info: Dict[str, Any]) -> None:
    """Print the evaluation details in a clean, structured format."""

    print("\U0001f4ca Detailed Info:")
    print()
    print("\u2714 Required Points Covered:")
    print_list_items(info["matched_required"])
    print()
    print("\u274c Missing Required Points:")
    print_list_items(info["missing_required"])
    print()
    print("\u2795 Optional Improvements Used:")
    print_list_items(info["matched_optional"])
    print()
    print("\u2764\ufe0f Empathy Detected:")
    print_list_items(info["matched_empathy"])
    print()
    print("\U0001f4cf Word Count:")
    print(f"  * {info['word_count']}")
    print()
    print_expected_keywords(info["expected_keywords"])


def print_demo_episode(
    level: str,
    episode_number: int,
    observation: str,
    action: str,
    result: Dict[str, Any],
) -> None:
    """Print one full interaction in a clean, beginner-friendly format."""

    info = result["info"]
    missing_required = info["missing_required"]
    matched_optional = info["matched_optional"]
    matched_empathy = info["matched_empathy"]
    word_count = int(info["word_count"]) if info.get("word_count") else 0

    summary_lines = [
        (
            "\u2714 Covered all required points"
            if not missing_required
            else "\u274c Missing some required points"
        ),
        (
            "\u2714 Included helpful extra details"
            if matched_optional
            else "\u274c Missing optional improvements"
        ),
        (
            "\u2714 Good polite tone"
            if matched_empathy
            else "\u274c Lacks polite/empathy tone"
        ),
        (
            "\u2714 Response length is good"
            if word_count >= 8
            else "\u274c Response is too short"
        ),
    ]

    scenario_id = f"{level.upper()}-{episode_number}"

    print("--------------------------------------------------")
    print()
    print(f"Scenario ID: {scenario_id}")
    print()
    print(f"\U0001f464 Customer: {observation}")
    print()
    print(f"\U0001f916 AI Response: {action}")
    print()
    print(f"\u2b50 Reward: {result['reward']:.3f}")
    print()
    print("\U0001f9e0 Evaluation Summary:")
    for line in summary_lines:
        print(line)
    print()
    print_detailed_info(info)


def print_level_summary(level: str, rewards: list[float]) -> None:
    """Print a compact reward summary for one difficulty level."""

    if not rewards:
        average_reward = 0.01
        best_reward = 0.01
        worst_reward = 0.01
    else:
        safe_rewards = [clamp_score(reward) for reward in rewards]
        average_reward = clamp_score(sum(safe_rewards) / len(safe_rewards))
        best_reward = clamp_score(max(safe_rewards))
        worst_reward = clamp_score(min(safe_rewards))

    print(f"\U0001f4ca Level Summary ({level.upper()}):")
    print()
    print(f"* Average Reward: {average_reward:.3f}")
    print(f"* Best Reward: {best_reward:.3f}")
    print(f"* Worst Reward: {worst_reward:.3f}")
    print()


def run_baseline_mode(
    env: SmartCustomerSupportEnv,
    policies: Dict[str, Callable[[str], str]],
) -> None:
    """Run the compact evaluation summary used for scoring and comparison."""

    print("Smart Customer Support Environment - Baseline Mode")
    print("-" * 56)
    print(f"Episodes per level: {BASELINE_EPISODES}")
    print()

    for policy_name, policy in policies.items():
        print(f"Policy: {policy_name}")
        for level in ("easy", "medium", "hard"):
            average_reward = run_policy(env, level, policy, BASELINE_EPISODES)
            print(f"  {level:<6} average reward: {average_reward:.3f}")
        print()


def run_demo_mode(
    env: SmartCustomerSupportEnv,
    policy_name: str,
    policy: Callable[[str], str],
) -> None:
    """Run a few episodes and print the full conversation and evaluation details."""

    print("Smart Customer Support Environment - Demo Mode")
    print("-" * 52)
    print(f"Policy: {policy_name}")
    print(f"Episodes per level: {DEMO_EPISODES}")
    print()

    for level in ("easy", "medium", "hard"):
        rewards: list[float] = []

        print(f"==================== {level.upper()} ====================")
        print()
        for episode_number in range(1, DEMO_EPISODES + 1):
            observation = env.reset(level)
            action = policy(observation)
            result = env.step(action)
            result["reward"] = clamp_score(result["reward"])
            rewards.append(result["reward"])
            print_demo_episode(level, episode_number, observation, action, result)
            print()
        print_level_summary(level, rewards)
        print()


def main() -> None:
    configure_output_encoding()
    random.seed(42)
    env = SmartCustomerSupportEnv(seed=42)
    policies: Dict[str, Callable[[str], str]] = {
        "random": random_agent,
        "rule_based": rule_based_agent,
    }

    if MODE == "baseline":
        run_baseline_mode(env, policies)
        return

    if MODE == "demo":
        demo_policy = policies[DEMO_POLICY_NAME]
        run_demo_mode(env, DEMO_POLICY_NAME, demo_policy)
        return

    valid_modes = ["baseline", "demo"]
    raise ValueError(f"Invalid MODE '{MODE}'. Choose one of: {valid_modes}")


if __name__ == "__main__":
    main()
