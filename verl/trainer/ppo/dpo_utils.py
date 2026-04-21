"""Utilities for online DPO training with online critique generation.

Sequence per row (B3 ordering):
  [sys][motion_prompt+obs]         <- raw_prompt from RMDataset (emit_components=True)
  [motion_response_gen]            <- rollout A: current policy generates initial motion
  [annotated_obs + critic_prompt]  <- rendered annotation + critic_prompt.txt template
  [critique_gen]                   <- rollout B: current policy generates critique
  [refine_prompt]                  <- static template from config
  [chosen | rejected motion]       <- offline-ranked pair from dataset; only these tokens scored

response_mask covers only the final motion tokens.  Critique and initial motion tokens
are context only (excluded from DPO logp sum).
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from cotnav.core.format import format_prompt as format_prompt_cotnav, text_to_llava
from cotnav.utils.draw import draw_polyline
from cotnav.prompts.interface import OutputFormat, parse_and_unify
from verl.models.transformers.qwen3_vl import get_rope_index
from verl.protocol import DataProto


class DPODataCollator:
    """Collate RMDataset emit_components=True samples into per-key lists.

    No padding or tensor ops — full sequences are built in build_dpo_batch
    after both online rollout stages complete.
    """

    def __call__(self, batch: list[dict]) -> dict:
        keys = batch[0].keys()
        return {k: [s[k] for s in batch] for k in keys}


# ---------------------------------------------------------------------------
# Per-sample helpers
# ---------------------------------------------------------------------------

def _build_annotated_critic_msg(
    motion_response_text: str,
    multi_modal_data: dict,
    critic_prompt_msg: dict,
) -> dict:
    """Return the critic user turn with a rendered annotated-path image.

    Replicates the per-sample logic of build_rm_raw_prompt (prompt_utils.py)
    without requiring the full non_tensor_dict wrapper.

    Args:
        motion_response_text: Raw text output from rollout A.
        multi_modal_data:     {"video": [(tensor, meta), ...]} or {"image": [PIL, ...]}.
        critic_prompt_msg:    Pre-loaded text_to_llava message from critic_prompt.txt.

    Returns:
        A copy of critic_prompt_msg with an annotated image prepended to content.
    """
    # Obtain base image: last frame of video or first image
    if "video" in multi_modal_data and multi_modal_data["video"]:
        video_entry = multi_modal_data["video"][0]
        video_tensor = video_entry[0] if isinstance(video_entry, tuple) else video_entry
        last_frame = video_tensor[-1]  # [3, H, W] from process_video
        last_frame = last_frame / 255.0 if last_frame.max() > 1.0 else last_frame
        base_img = to_pil_image(last_frame)
    elif "image" in multi_modal_data and multi_modal_data["image"]:
        img = multi_modal_data["image"][0]
        base_img = img if isinstance(img, Image.Image) else to_pil_image(img)
    else:
        raise ValueError("multi_modal_data must contain 'video' or 'image' entries.")

    # Parse trajectory for polyline overlay
    trace = None
    try:
        payload = parse_and_unify(motion_response_text, OutputFormat.TRAJECTORY_V1)
        if hasattr(payload, "unified") and "trajectory" in payload.unified:
            trace = payload.unified["trajectory"]
    except (ValueError, TypeError):
        pass

    if trace is not None:
        ann_arr = draw_polyline(
            trace, np.array(base_img), line_thickness=2, dot_radius=3, color=(255, 255, 51)
        )
        ann_img = Image.fromarray(ann_arr)
        motion_str = "[" + ", ".join(f"[{pt[0]:.3f}, {pt[1]:.3f}]" for pt in trace) + "]"
    else:
        ann_img = base_img
        motion_str = "[]"

    msg = copy.deepcopy(critic_prompt_msg)
    msg["content"][0]["text"] = format_prompt_cotnav(
        msg["content"][0]["text"], motion_str=motion_str
    )
    msg["content"].insert(0, {"type": "image", "image": ann_img})
    return msg


def _apply_chat_template_kw(processor, base_kwargs: dict) -> dict:
    kw = dict(base_kwargs)
    if processor.__class__.__name__ == "Qwen3VLProcessor":
        kw.setdefault("do_sample_frames", False)
    return kw


def _tokenize_sequence(
    messages: list[dict],
    processor,
    apply_chat_template_kwargs: dict,
    max_length: int,
    truncation: str = "left",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Tokenize a full message list.

    Returns:
        input_ids:      (T,)
        attention_mask: (T,)
        position_ids:   (4, T) for Qwen3-VL mrope, else (T,)
        mm_inputs:      remaining dict keys from apply_chat_template (image_grid_thw, etc.)
    """
    kw = _apply_chat_template_kw(processor, apply_chat_template_kwargs)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **kw,
    )
    inputs = dict(inputs)
    input_ids = inputs.pop("input_ids")[0]
    attention_mask = inputs.pop("attention_mask")[0]

    # Left-truncate to preserve the motion response at the tail
    seq_len = input_ids.shape[0]
    if seq_len > max_length:
        if truncation == "left":
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

    # Position IDs: use mrope for Qwen3-VL, plain arange otherwise
    has_qwen_vis = (
        processor is not None
        and hasattr(processor, "image_processor")
        and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
    )
    if has_qwen_vis:
        vision_pos = get_rope_index(
            processor,
            input_ids=input_ids,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            attention_mask=attention_mask,
        )  # (3, T)
        text_pos = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)  # (1, T)
        position_ids = torch.cat((text_pos, vision_pos), dim=0)  # (4, T)
    else:
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)  # (T,)

    return input_ids, attention_mask, position_ids, inputs


def _prefix_token_length(
    shared_messages: list[dict],
    processor,
    apply_chat_template_kwargs: dict,
) -> int:
    """Number of tokens in shared_messages + generation-prompt suffix.

    This equals the index at which the final assistant (motion) tokens begin
    in the full chosen/rejected sequence.
    """
    kw = _apply_chat_template_kw(processor, apply_chat_template_kwargs)
    ids = processor.apply_chat_template(
        shared_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        **kw,
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return int(ids.shape[-1])


# ---------------------------------------------------------------------------
# Main batch builder
# ---------------------------------------------------------------------------

def build_dpo_batch(
    batch_components: dict,
    motion_responses: list[str],
    critique_responses: list[str],
    critic_prompt_msg: dict,
    refine_prompt_msg: dict,
    processor,
    tokenizer,
    max_length: int,
    apply_chat_template_kwargs: Optional[dict] = None,
    extra_infos: Optional[list[dict]] = None,
) -> DataProto:
    """Build the interleaved DPO batch from dataset components and online rollouts.

    For each row constructs the full B3 sequence, tokenizes chosen and rejected
    sides, computes response_mask on motion tokens only, pads the pair to equal
    length, then interleaves across the batch: [c0, r0, c1, r1, ...].

    Args:
        batch_components:    Output of DPODataCollator — lists keyed by raw_prompt,
                             multi_modal_data, chosen_motion_text, rejected_motion_text.
        motion_responses:    Generated initial motion text per row (rollout A).
        critique_responses:  Generated critique text per row (rollout B).
        critic_prompt_msg:   Pre-loaded text_to_llava message from critic_prompt.txt.
        refine_prompt_msg:   Pre-loaded text_to_llava message from motion_refine_prompt.txt.
        processor:           HF processor (Qwen3VLProcessor or tokenizer).
        tokenizer:           HF tokenizer (for pad_token_id).
        max_length:          Maximum sequence length; left-truncated if exceeded.
        apply_chat_template_kwargs: Extra kwargs forwarded to apply_chat_template.
        extra_infos:         Optional per-row dicts with semantic_goal / vgoal for
                             filling refine_prompt template placeholders.

    Returns:
        DataProto with:
          batch tensors — input_ids (2N,T), attention_mask (2N,T),
                          position_ids (2N,4,T) or (2N,T), response_mask (2N,T),
                          pair_loss_mask (2N,)
          non_tensor_batch — multi_modal_inputs (2N,)
    """
    apply_chat_template_kwargs = apply_chat_template_kwargs or {}
    raw_prompts = batch_components["raw_prompt"]
    mm_data_list = batch_components["multi_modal_data"]
    chosen_texts = batch_components["chosen_motion_text"]
    rejected_texts = batch_components["rejected_motion_text"]
    N = len(raw_prompts)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    all_input_ids: list[torch.Tensor] = []
    all_attn_masks: list[torch.Tensor] = []
    all_pos_ids: list[torch.Tensor] = []
    all_resp_masks: list[torch.Tensor] = []
    all_pair_loss_masks: list[float] = []
    all_mm_inputs: list[dict] = []

    for i in range(N):
        raw_prompt: list[dict] = list(raw_prompts[i])
        mm_data: dict = mm_data_list[i]
        motion_resp: str = motion_responses[i]
        critique_resp: str = critique_responses[i]
        chosen_text: str = chosen_texts[i]
        rejected_text: str = rejected_texts[i]

        # ---- 1. Build annotated critic turn ----
        critic_msg = _build_annotated_critic_msg(motion_resp, mm_data, critic_prompt_msg)

        # ---- 2. Format refine prompt (fill goal/start tokens if extra_info present) ----
        refine_msg = copy.deepcopy(refine_prompt_msg)
        if extra_infos is not None:
            extra = extra_infos[i]
            language_goal = extra.get("semantic_goal", "")
            gt_start = extra.get("vgoal", "")
            if isinstance(gt_start, (list, np.ndarray)):
                arr = np.array(gt_start, dtype=np.float64).flatten()
                gt_start = f"[{arr[0]:.3f}, {arr[1]:.3f}]" if len(arr) >= 2 else str(gt_start)
            refine_msg["content"][0]["text"] = format_prompt_cotnav(
                refine_msg["content"][0]["text"],
                language_goal=str(language_goal),
                motion_start=str(gt_start),
            )

        # ---- 3. Assemble shared prefix (before the final motion token) ----
        shared_messages = (
            raw_prompt
            + [text_to_llava(motion_resp, role="assistant")]
            + [critic_msg]
            + [text_to_llava(critique_resp, role="assistant")]
            + [refine_msg]
        )

        # ---- 4. Full chosen / rejected sequences ----
        chosen_messages = shared_messages + [text_to_llava(chosen_text, role="assistant")]
        rejected_messages = shared_messages + [text_to_llava(rejected_text, role="assistant")]

        # ---- 5. Compute response start index ----
        prefix_len = _prefix_token_length(shared_messages, processor, apply_chat_template_kwargs)

        # ---- 6. Tokenize both sides ----
        c_ids, c_attn, c_pos, c_mm = _tokenize_sequence(
            chosen_messages, processor, apply_chat_template_kwargs, max_length
        )
        r_ids, r_attn, r_pos, r_mm = _tokenize_sequence(
            rejected_messages, processor, apply_chat_template_kwargs, max_length
        )

        # ---- 7. Response mask: 1 only for final motion tokens ----
        c_resp = torch.zeros(c_ids.shape[0], dtype=torch.float32)
        r_resp = torch.zeros(r_ids.shape[0], dtype=torch.float32)
        # Clamp in case left-truncation ate into the prefix
        c_resp[min(prefix_len, c_ids.shape[0]):] = 1.0
        r_resp[min(prefix_len, r_ids.shape[0]):] = 1.0
        # Mask out pad tokens
        c_resp = c_resp * c_attn.float()
        r_resp = r_resp * r_attn.float()

        pair_valid = float((c_resp.sum() > 0) and (r_resp.sum() > 0))

        # ---- 8. Pad chosen / rejected to the same length within the pair ----
        pair_max = max(c_ids.shape[0], r_ids.shape[0])

        def _rpad(t: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
            pad = length - t.shape[-1]
            return t if pad <= 0 else F.pad(t, (0, pad), value=value)

        c_ids  = _rpad(c_ids,  pair_max, pad_id)
        r_ids  = _rpad(r_ids,  pair_max, pad_id)
        c_attn = _rpad(c_attn, pair_max, 0)
        r_attn = _rpad(r_attn, pair_max, 0)
        c_resp = _rpad(c_resp, pair_max, 0.0)
        r_resp = _rpad(r_resp, pair_max, 0.0)
        c_pos  = _rpad(c_pos,  pair_max, 0)
        r_pos  = _rpad(r_pos,  pair_max, 0)

        # ---- Interleave [chosen, rejected] per pair ----
        all_input_ids.extend([c_ids, r_ids])
        all_attn_masks.extend([c_attn, r_attn])
        all_pos_ids.extend([c_pos, r_pos])
        all_resp_masks.extend([c_resp, r_resp])
        all_pair_loss_masks.extend([pair_valid, pair_valid])
        all_mm_inputs.extend([c_mm, r_mm])

    # ---- Stack to (2N, T) padded to global max length across all pairs ----
    global_max = max(t.shape[-1] for t in all_input_ids)

    def _stack_rpad(tensors: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
        padded = [F.pad(t, (0, global_max - t.shape[-1]), value=pad_value) for t in tensors]
        return torch.stack(padded, dim=0)

    input_ids    = _stack_rpad(all_input_ids,  pad_id)   # (2N, T)
    attention_mask = _stack_rpad(all_attn_masks, 0.0)    # (2N, T)
    response_mask  = _stack_rpad(all_resp_masks, 0.0)    # (2N, T)
    position_ids   = _stack_rpad(all_pos_ids,   0.0)     # (2N, T) or (2N, 4, T)
    pair_loss_mask = torch.tensor(all_pair_loss_masks, dtype=torch.float32)  # (2N,)

    return DataProto.from_dict(
        tensors={
            "input_ids":     input_ids,
            "attention_mask": attention_mask,
            "position_ids":  position_ids,
            "response_mask": response_mask,
            "pair_loss_mask": pair_loss_mask,
        },
        non_tensors={
            "multi_modal_inputs": np.array(all_mm_inputs, dtype=object),
        },
    )