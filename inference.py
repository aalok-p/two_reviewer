import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from env_server import KernelOptimization_env, TASKS, grade_episode
from models import Action

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.5")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
TASK_NAME = os.getenv("TASK_ID")
BENCHMARK = "kernel_optimization"


def one_line(text: str) -> str:
    return " ".join((text or "").split())


def extract_code(text: str) -> str:
    if "```" not in text:
        return text
    start = text.find("```")
    end = text.rfind("```")
    chunk = text[start + 3 : end]
    if chunk.startswith("cuda") or chunk.startswith("cpp"):
        return chunk.split("\n", 1)[1]
    return chunk


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = one_line(error) if error else "null"
    done_val = str(done).lower()
    action_val = one_line(action)
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def fallback_action(observation: dict) -> Action:
    # Deterministic, compile-safe fallback when remote model is unavailable.
    return Action(optimized_code=observation["current_best_code"], strategy="fallback")


def choose_action(client: Optional[OpenAI], observation: dict) -> Action:
    if client is None:
        return fallback_action(observation)
    prompt = (
        "Optimize this CUDA kernel.\n"
        f"Task: {observation['task_name']}\n"
        f"Pending checks: {observation['pending_checks']}\n"
        f"Current code:\n{observation['current_best_code']}\n"
        "Return only optimized CUDA code."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a CUDA optimization expert. Return code only."},
                {"role": "user", "content": prompt},
            ],
        )
        content = (completion.choices[0].message.content or "").strip()
        code = extract_code(content).strip() or observation["current_best_code"]
        return Action(optimized_code=code, strategy="llm_proposed")
    except Exception:
        return fallback_action(observation)


def run_episode(client: Optional[OpenAI], task_id: str) -> None:
    env = KernelOptimization_env()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        obs = env.reset(task_id=task_id)["observation"]
        done = False
        while not done:
            action = choose_action(client, obs)
            step_result = env.step(action)
            done = step_result.done
            obs = step_result.observation.model_dump()
            reward = step_result.reward.value
            rewards.append(reward)
            steps_taken = obs["step_count"]
            log_step(step=steps_taken, action=action.optimized_code, reward=reward, done=done, error=None)

        score = grade_episode(
            task_id,
            env.state.completed_checks,
            env.state.best_speedup,
            env.state.step_count,
            env.state.max_steps,
        )
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.1
    except Exception as exc:
        log_step(step=max(1, steps_taken + 1), action="error", reward=0.0, done=True, error=str(exc))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> int:
    client: Optional[OpenAI] = None
    if API_KEY:
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        except Exception:
            client = None

    if TASK_NAME and TASK_NAME in TASKS:
        task_ids = [TASK_NAME]
    else:
        task_ids = list(TASKS.keys())

    for task_id in task_ids:
        run_episode(client, task_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
