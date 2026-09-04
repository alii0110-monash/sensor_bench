"""Chat-sequence construction with sensor-token span injection + batched greedy
decode. Both conditions (with/without sensor tokens) share identical text, so
prompt lengths are uniform per condition — no padding needed for decoding."""
from __future__ import annotations
import torch

SYSTEM = "You are a sensor activity recognition assistant."
QUESTION = "Based on the sensor data, what is the person doing? Answer with the activity name only."
IM_END = "<|im_end|>"


def _pre_text() -> str:
    return f"<|im_start|>system\n{SYSTEM}{IM_END}\n<|im_start|>user\n"


def _post_text(question: str | None = None) -> str:
    return f"{question or QUESTION}{IM_END}\n<|im_start|>assistant\n"


def encode_prompt_ids(tokenizer, question: str | None = None):
    pre = tokenizer(_pre_text(), add_special_tokens=False)["input_ids"]
    post = tokenizer(_post_text(question), add_special_tokens=False)["input_ids"]
    return pre, post


def encode_target_ids(tokenizer, answer: str) -> list:
    return tokenizer(answer + IM_END, add_special_tokens=False)["input_ids"]


def _embed_ids(ids, emb_layer, device):
    t = torch.tensor(ids, dtype=torch.long, device=device)
    return emb_layer(t)


def build_train_sample(emb_layer, pre_ids, post_ids, target_ids,
                       sensor_embeds, device, pad_emb):
    """Returns (embeds (L,H), labels (L,)) for one sample; sensor span sits
    between pre and post; loss only on target ids."""
    parts = [_embed_ids(pre_ids, emb_layer, device)]
    n_sensor = 0
    if sensor_embeds is not None:
        parts.append(sensor_embeds)
        n_sensor = sensor_embeds.shape[0]
    parts.append(_embed_ids(post_ids, emb_layer, device))
    parts.append(_embed_ids(target_ids, emb_layer, device))
    embeds = torch.cat(parts, dim=0)
    labels = torch.full((embeds.shape[0],), -100, dtype=torch.long, device=device)
    labels[-len(target_ids):] = torch.tensor(target_ids, dtype=torch.long, device=device)
    return embeds, labels, n_sensor


def pad_train_batch(samples_embeds, pad_emb):
    """Right-pad variable-length (embeds, labels) into (B,L,H)/(B,L)/(B,L)."""
    lens = [e.shape[0] for e, _lab, _ns in samples_embeds]
    L = max(lens)
    H = samples_embeds[0][0].shape[-1]
    B = len(samples_embeds)
    device = samples_embeds[0][0].device
    embeds = pad_emb.unsqueeze(0).unsqueeze(0).expand(B, L, H).contiguous()
    labels = torch.full((B, L), -100, dtype=torch.long, device=device)
    attn = torch.zeros(B, L, dtype=torch.long, device=device)
    for i, (e, lab, _ns) in enumerate(samples_embeds):
        embeds[i, :e.shape[0]] = e
        labels[i, :lab.shape[0]] = lab
        attn[i, :e.shape[0]] = 1
    return embeds, attn, labels


def build_prompt_embeds(emb_layer, pre_ids, post_ids, sensor_embeds, device):
    """Eval prompt embeds (B, L, H); uniform length within a condition."""
    pre = _embed_ids(pre_ids, emb_layer, device)
    post = _embed_ids(post_ids, emb_layer, device)
    parts = [pre]
    if sensor_embeds is not None:
        parts.append(sensor_embeds)
    parts.append(post)
    return torch.cat(parts, dim=0).unsqueeze(0)


@torch.no_grad()
def batched_prompt_embeds(emb_layer, pre_ids, post_ids, sensor_embeds_b, device):
    """Stack per-sample prompts; sensor_embeds_b None or (B, S, H)."""
    prompts = []
    B = 1 if sensor_embeds_b is None else sensor_embeds_b.shape[0]
    for i in range(B):
        se = None if sensor_embeds_b is None else sensor_embeds_b[i]
        prompts.append(build_prompt_embeds(emb_layer, pre_ids, post_ids, se, device))
    return torch.cat(prompts, dim=0)


@torch.no_grad()
def greedy_decode(model, prompt_embeds, tokenizer, max_new_tokens: int = 12) -> list:
    """Manual greedy decode with KV cache; uniform-length prompts assumed.
    Returns list of decoded strings (cut at <|im_end|>)."""
    device = prompt_embeds.device
    B, L, _ = prompt_embeds.shape
    emb_layer = model.get_input_embeddings()
    attn = torch.ones(B, L, dtype=torch.long, device=device)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    out = model(inputs_embeds=prompt_embeds, attention_mask=attn, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1].argmax(-1)
    gen = [next_tok]
    for _ in range(max_new_tokens - 1):
        attn = torch.cat([attn, torch.ones(B, 1, dtype=torch.long, device=device)], dim=1)
        cur = emb_layer(gen[-1]).unsqueeze(1)
        out = model(inputs_embeds=cur, attention_mask=attn, past_key_values=past,
                    use_cache=True)
        past = out.past_key_values
        gen.append(out.logits[:, -1].argmax(-1))
    ids = torch.stack(gen, dim=1).tolist()
    texts = []
    for seq in ids:
        if im_end_id in seq:
            seq = seq[:seq.index(im_end_id)]
        texts.append(tokenizer.decode(seq, skip_special_tokens=True).strip())
    return texts
