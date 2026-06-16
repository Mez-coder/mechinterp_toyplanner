"""Agents that produce an action string given the message history.

HumanAgent  -- you type the action (for --human roleplay).
ModelAgent  -- loads the HF model, generates, optionally captures activations.

torch/transformers are imported lazily inside ModelAgent so --human needs neither.
"""
from __future__ import annotations
import os


class HumanAgent:
    is_model = False

    def act(self, messages, capture_path=None):
        # show the latest environment turn, then read one action line
        print("\n" + "=" * 70)
        print(messages[-1]["content"])
        print("=" * 70)
        text = input("your action > ").strip()
        return text, {"source": "human"}


class ModelAgent:
    is_model = True

    def __init__(self, cfg):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.bfloat16, device_map=cfg.device)
        self.model.eval()
        self.device = cfg.device

    def _build_inputs(self, messages):
        # Gemma has no system role -> fold system text into the first user turn.
        msgs, sys_txt = [], None
        for m in messages:
            if m["role"] == "system":
                sys_txt = m["content"]; continue
            if sys_txt and m["role"] == "user":
                m = {"role": "user", "content": sys_txt + "\n\n" + m["content"]}
                sys_txt = None
            msgs.append(m)
        return self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)

    def act(self, messages, capture_path=None):
        import torch
        ids = self._build_inputs(messages)
        prompt_len = ids.shape[1]
        gen = self.model.generate(
            ids, max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.temperature > 0, temperature=self.cfg.temperature,
            pad_token_id=self.tok.eos_token_id)
        full = gen                                   # (1, prompt+resp)
        text = self.tok.decode(full[0, prompt_len:], skip_special_tokens=True).strip()
        meta = {"source": "model", "prompt_len": int(prompt_len),
                "resp_len": int(full.shape[1] - prompt_len),
                "temperature": self.cfg.temperature}
        if capture_path and self.cfg.capture:
            from .recorder import capture_and_save
            capture_and_save(self.model, full, prompt_len, capture_path,
                             tokens=self.cfg.capture_tokens, dtype=self.cfg.capture_dtype)
            meta["activations"] = os.path.basename(capture_path)
        return text, meta
