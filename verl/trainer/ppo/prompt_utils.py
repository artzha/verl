# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prompt formatting and VLM input preparation for rollout generation.

Used by AgentLoopWorker to keep generate_motion_sequences and generate_judge_sequences
thin: encoding/decoding and prompt updates are delegated here.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info

from cotnav.core.constants import (
    MOTION_END_TOKEN,
    MOTION_GOAL_TOKEN,
    MOTION_START_TOKEN,
    LANGUAGE_GOAL_TOKEN,
)
from cotnav.core.format import format_prompt as format_prompt_cotnav
from cotnav.core.format import text_to_llava
from cotnav.models.vlms.interface import OutputFormat, parse_and_unify
from cotnav.utils.draw import draw_polyline


def prepare_inputs_for_vllm(
    messages: list[dict],
    processor: Any,
) -> dict[str, Any]:
    """Build VLM-ready prompt text and multi_modal_data from a message list.

    Args:
        messages: List of chat messages (e.g. LLaVA-style with content).
        processor: HF processor with apply_chat_template and image_processor.

    Returns:
        Dict with "prompt" (str) and "multi_modal_data" (dict with "image" / "video").
    """
    assert processor is not None, "processor is required to build prompts from messages"
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, _ = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data: dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {"prompt": text, "multi_modal_data": mm_data}


def format_prompt(
    non_tensor_dict: dict,
    i: int,
    prompt_msg: dict,
    prev_message: Optional[list] = None,
) -> dict:
    """Fill in goal tokens and optional previous motion into a prompt message.

    Expects extra_info[i] to contain vgoal, semantic_goal; prev_message can
    contain previous assistant content (e.g. motion trajectory) for templating.

    Args:
        non_tensor_dict: non_tensor_batch with at least "extra_info".
        i: Sample index.
        prompt_msg: Message dict with content[0]["text"] (template).
        prev_message: Optional previous message content for motion placeholder.

    Returns:
        prompt_msg with content[0]["text"] updated (in place and returned).
    """
    extra = non_tensor_dict["extra_info"][i]
    vgoal = extra.get("vgoal", [])
    semantic_goal = extra.get("semantic_goal", "")
    vgoal_str = f"[{vgoal[0][0]:.3f}, {vgoal[0][1]:.3f}]" if len(vgoal) > 0 else "[N/A, N/A]"

    prompt_template = prompt_msg["content"][0]["text"]
    prompt = format_prompt_cotnav(
        prompt_template,
        visual_goal=vgoal_str,
        language_goal=semantic_goal,
    )

    if prev_message is not None:
        try:
            payload = parse_and_unify(prev_message[0]["text"], OutputFormat.TRAJECTORY_V1)
            trace_pts = payload.unified["trajectory"]
            trace_pts_str = "[" + ", ".join([f"[{pt[0]:.3f}, {pt[1]:.3f}]" for pt in trace_pts]) + "]"
            prompt = format_prompt_cotnav(
                prompt,
                motion_str=trace_pts_str,
            )
        except (ValueError, TypeError):
            pass
    prompt_msg["content"][0]["text"] = prompt
    return prompt_msg


def update_prompt_after_response(
    non_tensor_dict: dict,
    i: int,
    response_text: str,
    processor: Any,
    prompts_registry: dict[str, dict],
    *,
    prefill: Optional[str] = None,
    postfill: Optional[str] = None,
    ann: Optional[Image.Image] = None,
    format_prompt_fn: Optional[Callable[..., dict]] = None,
) -> None:
    """Append prefill, assistant response, and postfill to raw_prompt and update full_prompts.

    Updates non_tensor_dict in place: raw_prompt[i], full_prompts[i], and
    optionally multi_modal_data[i]["image"] if ann is provided.

    Args:
        non_tensor_dict: non_tensor_batch with raw_prompt, full_prompts, optional multi_modal_data.
        i: Sample index.
        response_text: Assistant response text to append.
        processor: Used for apply_chat_template.
        prompts_registry: Map prompt name -> LLaVA-style message (e.g. from prompt_paths).
        prefill: Key in prompts_registry for message to add before the response.
        postfill: Key in prompts_registry for message to add after the response.
        ann: Optional image to prepend to postfill message content.
        format_prompt_fn: Optional (non_tensor_dict, i, prompt_msg, prev_message) -> prompt_msg.
    """
    messages = list(non_tensor_dict["raw_prompt"][i])
    if messages and isinstance(messages[0], list):
        raise RuntimeError(
            f"raw_prompt[{i}] is nested (list of lists); expected list of dicts."
        )

    if prefill is not None:
        assert prefill in prompts_registry, f"Unknown prefill key {prefill}"
        prefill_msg = copy.deepcopy(prompts_registry[prefill])
        if format_prompt_fn is not None:
            prefill_msg = format_prompt_fn(
                non_tensor_dict, i, prefill_msg,
                prev_message=messages[-1]["content"] if messages else None,
            )
        if not messages or messages[-1] != prefill_msg:
            messages.append(copy.deepcopy(prefill_msg))

    messages.append(text_to_llava(response_text, role="assistant"))

    if postfill is not None:
        assert postfill in prompts_registry, f"Unknown postfill key {postfill}"
        post_msg = copy.deepcopy(prompts_registry[postfill])
        if format_prompt_fn is not None:
            post_msg = format_prompt_fn(
                non_tensor_dict, i, post_msg,
                prev_message=messages[-1]["content"] if messages else None,
            )
        if ann is not None:
            post_msg["content"].insert(0, {"type": "image", "image": ann})
        messages.append(post_msg)

    assert processor is not None, "processor is required to update prompts"
    new_full_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    non_tensor_dict["raw_prompt"][i] = messages
    non_tensor_dict["full_prompts"][i] = new_full_prompt
    if ann is not None and "multi_modal_data" in non_tensor_dict:
        non_tensor_dict["multi_modal_data"][i]["image"].append(ann)


def draw_path(new_non_tensor: dict, cur_idx: int = 0) -> dict[str, Any]:
    """Parse motion response strings and draw polylines on the source image.

    Expects new_non_tensor["motion_response"] (list of lists or list of strings).
    Expects new_non_tensor["multi_modal_data"][i]["image"] to be a list of images.

    Returns:
        Dict with "trace" (list of trajectory or None) and "image" (list of PIL Image or None).
    """
    assert "motion_response" in new_non_tensor
    motion_response = new_non_tensor["motion_response"]
    images = []
    traces = []
    for i, trace_list in enumerate(motion_response):
        trace_str = (
            trace_list[0]
            if isinstance(trace_list, (list, tuple, np.ndarray))
            else trace_list
        )
        try:
            payload = parse_and_unify(trace_str, OutputFormat.TRAJECTORY_V1)
        except (ValueError, TypeError):
            payload = None
        if hasattr(payload, "unified") and "trajectory" in payload.unified:
            trace = payload.unified["trajectory"]
        else:
            trace = None
        traces.append(trace)
        cur_img = new_non_tensor["multi_modal_data"][i]["image"][cur_idx]
        if trace is None:
            images.append(cur_img)
            continue
        ann_img = draw_polyline(
            trace, np.array(cur_img), line_thickness=2, dot_radius=3, color=(255, 255, 51)
        )
        images.append(Image.fromarray(ann_img))
    return {"trace": traces, "image": images}
