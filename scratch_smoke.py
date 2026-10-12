"""Smoke test: judge path (fast model) + answer path (thinking model)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentic_rag.config import Config
from agentic_rag.llm import NIMBrain, LLMOutputError

cfg = Config()
print(f"key set: {bool(cfg.nim_api_key)}")
print(f"answer model: {cfg.nim_model}  thinking: {cfg.nim_enable_thinking}  budget: {cfg.nim_reasoning_budget}")
print(f"judge model : {cfg.nim_judge_model or '(same as answer)'}")
brain = NIMBrain(cfg)

t = time.time()
try:
    data = brain._complete_json(
        "You are a test judge. Return ONLY a JSON object.",
        'Return {"status": "pass", "confidence": 0.95, "reason": "smoke ok"}.',
        model=brain.judge_model, thinking=False,
    )
    print(f"JUDGE path OK in {time.time() - t:.1f}s:", data)
except LLMOutputError as exc:
    print("JUDGE FAILED:", exc)
    sys.exit(1)

t = time.time()
try:
    data = brain._complete_json(
        "You are the ANSWER_AGENT. Return ONLY a JSON object.",
        'Answer "What is 2+2?" with {"stage": "answer_generation", "status": "pass", '
        '"confidence": 0.99, "answer": "2+2 equals 4.", "citations": [], '
        '"unanswered_parts": [], "reason": "smoke ok"}.',
        model=cfg.nim_model, thinking=cfg.nim_enable_thinking,
    )
    print(f"ANSWER path OK in {time.time() - t:.1f}s:", data)
except LLMOutputError as exc:
    print("ANSWER FAILED:", exc)
    sys.exit(1)
