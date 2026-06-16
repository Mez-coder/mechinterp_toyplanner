"""Activation capture: after the model generates an action, re-forward the full
sequence with output_hidden_states and save the residual stream at the assistant
token positions across ALL layers. The 'decision' token (last generated) is the
position where SET-vs-SUBMIT is committed -- flagged explicitly.

Saved npz per turn:
  acts   : (n_layers+1, n_positions, d_model)  [capture_dtype]
  positions     : absolute token indices captured
  decision_index: index into `positions` of the commit token
  token_ids     : the captured tokens
This module imports torch lazily so the --human path needs no GPU/torch.
"""
from __future__ import annotations
import os
import numpy as np


def capture_and_save(model, full_ids, prompt_len, path, tokens="assistant",
                     dtype="float16"):
    import torch
    with torch.no_grad():
        out = model(full_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states                      # tuple (L+1) of (1, seq, d_model)
    seq = full_ids.shape[1]
    if tokens == "decision":
        positions = [seq - 1]
    else:                                        # all assistant-generated tokens
        positions = list(range(prompt_len, seq))
    if not positions:
        positions = [seq - 1]
    idx = torch.tensor(positions, device=hs[0].device)
    stack = torch.stack([h[0].index_select(0, idx) for h in hs])   # (L+1, n_pos, d)
    np_dtype = np.float16 if dtype == "float16" else np.float32
    np.savez_compressed(
        path,
        acts=stack.to(torch.float32).cpu().numpy().astype(np_dtype),
        positions=np.array(positions),
        decision_index=len(positions) - 1,
        token_ids=full_ids[0, positions].cpu().numpy(),
    )
