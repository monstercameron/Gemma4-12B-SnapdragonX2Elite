"""fp16-safe eager attention for HTP. Two mathematically-identical reformulations that keep
intermediates inside fp16 range:

  1) q.k matmul: compute (q/C) @ k^T  then  * (scaling*C)  -> same result, but the matmul
     accumulator is C smaller, avoiding fp16 overflow when raw scores exceed 65504.
  2) softmax: explicitly subtract the row-max BEFORE softmax so exp() never sees a positive
     argument -> no overflow even if the HTP runs the kernel in fp16.

Gemma uses scaling=1.0 (no 1/sqrt(d)), so raw q.k can be large; both reformulations are exact.
"""
import torch
from torch import nn

C = 16.0  # q.k pre-scale (~sqrt(head_dim=256)); restores exactly via *(scaling*C)


def _repeat_kv(hidden_states, n_rep):
    b, h, s, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def safe_eager_attention(module, query, key, value, attention_mask,
                         dropout=0.0, scaling=None, softcap=None, **kwargs):
    if scaling is None:
        scaling = module.head_dim ** -0.5
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    # (1) overflow-safe scaled scores
    attn_weights = torch.matmul(query / C, key_states.transpose(2, 3)) * (scaling * C)
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # (2) explicit max-subtraction -> fp16-safe softmax
    attn_weights = attn_weights - attn_weights.amax(dim=-1, keepdim=True)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def apply(M):
    """Patch the eager attention used by gemma4_unified export."""
    M.eager_attention_forward = safe_eager_attention
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = safe_eager_attention
    except Exception:
        pass
    return M
