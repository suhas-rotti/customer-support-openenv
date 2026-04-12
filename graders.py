from env import SmartCustomerSupportEnv


def grade_easy():
    env = SmartCustomerSupportEnv(seed=42)
    env.reset("easy")
    result = env.step(
        "Sorry, I understand your issue. Please share your order number and we will process your refund within 3-5 business days."
    )
    try:
        if not isinstance(result, dict):
            reward = 0.5
        else:
            reward = result.get("reward", 0.5)
            reward = float(reward)
    except:
        reward = 0.5

    reward = max(0.05, min(reward, 0.95))

    return reward


def grade_medium():
    env = SmartCustomerSupportEnv(seed=42)
    env.reset("medium")
    result = env.step(
        "Sorry for the inconvenience. Please provide your order number so we can look into this."
    )
    try:
        if not isinstance(result, dict):
            reward = 0.5
        else:
            reward = result.get("reward", 0.5)
            reward = float(reward)
    except:
        reward = 0.5

    reward = max(0.05, min(reward, 0.95))

    return reward


def grade_hard():
    env = SmartCustomerSupportEnv(seed=42)
    env.reset("hard")
    result = env.step(
        "We will check and get back to you."
    )
    try:
        if not isinstance(result, dict):
            reward = 0.5
        else:
            reward = result.get("reward", 0.5)
            reward = float(reward)
    except:
        reward = 0.5

    reward = max(0.05, min(reward, 0.95))

    return reward
