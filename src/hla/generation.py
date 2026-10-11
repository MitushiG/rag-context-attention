"""Generator wrapper (vLLM) with greedy decoding and answer log-probabilities.

The mean token log-probability of the generated answer is the confidence signal used in
H7. exp(mean logprob) is the geometric-mean token probability, a number in (0, 1].
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class Generation:
    text: str
    mean_logprob: float
    n_tokens: int

    @property
    def confidence(self) -> float:
        return math.exp(self.mean_logprob) if self.n_tokens else 0.0


class Generator:
    def __init__(self, model: str = DEFAULT_MODEL, max_model_len: int = 32768,
                 gpu_memory_utilization: float = 0.90, seed: int = 0):
        from vllm import LLM

        self.llm = LLM(model=model, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, seed=seed)
        self.tokenizer = self.llm.get_tokenizer()
        self.n_calls = 0
        self.seconds = 0.0

    def chat_prompt(self, user: str, system: str | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, prompts: list[str], max_tokens: int = 32,
                 stop: list[str] | None = None) -> list[Generation]:
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=stop, logprobs=1)
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        self.seconds += time.perf_counter() - t0
        self.n_calls += len(prompts)
        res = []
        for o in outs:
            c = o.outputs[0]
            n = len(c.token_ids)
            cum = c.cumulative_logprob if c.cumulative_logprob is not None else float("-inf")
            res.append(Generation(text=c.text.strip(), mean_logprob=(cum / n) if n else float("-inf"), n_tokens=n))
        return res
