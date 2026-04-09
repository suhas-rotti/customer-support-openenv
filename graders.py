from env import SmartCustomerSupportEnv


def grade_easy():
    env = SmartCustomerSupportEnv(seed=42)
    env.reset("easy")
    result = env.step(
        "Sorry, I understand your issue. Please share your order number and we will process your refund within 3-5 business days."
    )
    reward = float(result["reward"])
    reward = float(f"{reward:.6f}")
    reward = max(0.02, min(reward, 0.98))
    return reward


def grade_medium():
    env = SmartCustomerSupportEnv(seed=42)
    env.reset("medium")
    result = env.step(
        "Sorry for the inconvenience. Please provide your order number so we can look into this."
    )
    reward = float(result["reward"])
    reward = float(f"{reward:.6f}")
    reward = max(0.02, min(reward, 0.98))
    return reward


def grade_hard():
    env = SmartCustomerSupportEnv(seed=42)
    env.reset("hard")
    result = env.step(
        "We will check and get back to you."
    )
    reward = float(result["reward"])
    reward = float(f"{reward:.6f}")
    reward = max(0.02, min(reward, 0.98))
    return reward
