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
import asyncio
import copy
import heapq
import time
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.experimental.reward_loop import RewardLoopWorker
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
    rollout_trace_op,
)
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class
from verl.trainer.ppo import prompt_utils as prompt_utils_ppo
from cotnav.core.format import text_to_llava

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, idx, server] for idx, server in enumerate(self.server_handles)]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        _, _, server = self.weighted_serveres[0]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=uuid4().hex,  # use new request_id for each turn
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        dataset_config: DictConfig,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (DictConfigWrap): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
            dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
            dataset_config (DictConfig): Dataset config.
        """
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.dataset_config = dataset_config
        self.apply_chat_template_kwargs = dataset_config.get("apply_chat_template_kwargs", {})
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Extract images and videos from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys "image" and "video".
        """
        multi_modal_data = {}
        if self.processor is not None:
            images, videos = await self.dataset_cls.process_vision_info(
                messages, image_patch_size=self.processor.image_processor.patch_size, config=self.dataset_config
            )
            if images is not None:
                multi_modal_data["image"] = images
            if videos is not None:
                multi_modal_data["video"] = videos

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            # split the videos and according metadatas
            if videos is not None:
                videos, video_metadatas = zip(*videos, strict=False)
                videos, video_metadatas = list(videos), list(video_metadatas)
                # Strip keys not accepted by VideoMetadata dataclass (e.g. do_sample_frames)
                _valid_video_metadata_keys = {"total_num_frames", "fps", "width", "height", "duration", "video_backend", "frames_indices"}
                video_metadatas = [
                    {k: v for k, v in m.items() if k in _valid_video_metadata_keys} if isinstance(m, dict) else m
                    for m in video_metadatas
                ]
            else:
                video_metadatas = None

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
        extra_server_handles: Optional[dict[str, list]] = None,
        extra_rollouts_list: Optional[list] = None,
    ):
        """Initialize agent loop worker.

        Args:
            config: YAML config.
            server_handles: OpenAI compatible LLM server actor handles (primary rollout).
            reward_router_address: reward router address.
            extra_server_handles: Optional dict mapping rollout name -> list of server handles.
            extra_rollouts_list: Optional list of extra rollout configs (each with name, model, rollout, prompt_paths).
        """
        self.config = config

        if not hasattr(self, "server_manager"):
            self.server_manager = AsyncLLMServerManager(config, server_handles)

        self.dataset_cls = get_dataset_class(config.data)
        self.reward_router_address = reward_router_address
        self.extra_managers = {}
        self.extra_tokenizers = {}
        self.extra_processors = {}
        self.extra_prompts = {}

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        extra_server_handles = extra_server_handles or {}
        extra_rollouts_list = extra_rollouts_list if extra_rollouts_list is not None else []
        for entry in extra_rollouts_list:
            name = entry.name
            handles = extra_server_handles.get(name)
            if not handles:
                continue
            self.extra_managers[name] = AsyncLLMServerManager(config, handles)
            model_cfg = entry.model
            local_path = copy_to_local(model_cfg.path)
            trust = getattr(model_cfg, "trust_remote_code", True)
            self.extra_tokenizers[name] = hf_tokenizer(local_path, trust_remote_code=trust)
            self.extra_processors[name] = hf_processor(local_path, trust_remote_code=trust)
            template = getattr(model_cfg, "custom_chat_template", None)
            if template is not None:
                if self.extra_processors[name] is not None:
                    self.extra_processors[name].chat_template = template
                self.extra_tokenizers[name].chat_template = template
            prompt_paths = getattr(entry, "prompt_paths", None) or {}
            prompts = {}
            for key, path in prompt_paths.items():
                with open(path, "r") as handle:
                    prompts[key] = text_to_llava(handle.read(), role="user")
            self.extra_prompts[name] = prompts

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        use_reward_loop = True if self.config.reward_model.use_reward_loop else None
        self.use_reward_loop = use_reward_loop
        if use_reward_loop and not hasattr(self, "reward_loop_worker"):
            self.reward_loop_worker = RewardLoopWorker.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(self.config, self.reward_router_address)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    def draw_path(self, new_non_tensor: dict, cur_idx: int = -1) -> dict:
        """Parse motion response strings and draw polylines on the source image."""
        return prompt_utils_ppo.draw_path(new_non_tensor, cur_idx=cur_idx)

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)
        response_role = self.config.actor_rollout_ref.rollout.get("response_role", None)
        if response_role in ("critic", "motion"):
            response_key = "critic_response" if response_role == "critic" else "motion_response"
            decoded = self.tokenizer.batch_decode(output.batch["responses"], skip_special_tokens=True)
            assert len(decoded) == len(output), (
                f"Decoded response rows {len(decoded)} must match output batch size {len(output)}"
            )

            pre_existing = batch.non_tensor_batch.get(response_key, None)
            if pre_existing is None:
                output.non_tensor_batch[response_key] = np.array([[resp] for resp in decoded], dtype=object)
            else:
                existing_list = pre_existing.tolist()
                assert len(existing_list) == len(decoded), (
                    f"Existing `{response_key}` rows {len(existing_list)} "
                    f"must match decoded rows {len(decoded)}"
                )
                assert all(isinstance(row, list) for row in existing_list), (
                    f"Expected `{response_key}` rows to be list[str] history."
                )
                output.non_tensor_batch[response_key] = np.array(
                    [prev + [resp] for prev, resp in zip(existing_list, decoded)], dtype=object
                )

        return output

    @tqbridge()
    async def generate_motion_sequences(self, batch: DataProto) -> DataProto:
        """Generate motion sequences via extra_rollouts['motion']."""
        if "motion" not in self.extra_managers:
            raise RuntimeError(
                "motion rollout is not initialized; add a 'motion' entry to config.actor_rollout_ref.extra_rollouts"
            )
        motion_manager = self.extra_managers["motion"]
        motion_tokenizer = self.extra_tokenizers["motion"]
        motion_processor = self.extra_processors.get("motion")
        motion_prompts = self.extra_prompts.get("motion", {})
        
        extra_list = OmegaConf.select(self.config.actor_rollout_ref, "extra_rollouts")
        motion_cfg = (
            next((e.rollout for e in extra_list if e.name == "motion"), None)
            if extra_list
            else None
        )
        if motion_cfg is None:
            motion_cfg = self.config.actor_rollout_ref.rollout
        prefill = batch.meta_info.get("prefill", None)
        postfill = batch.meta_info.get("postfill", None)
        sampling_params = {
            "temperature": getattr(motion_cfg, "temperature", 1.0),
            "top_p": getattr(motion_cfg, "top_p", 0.95),
            "repetition_penalty": getattr(motion_cfg, "repetition_penalty", 1.0),
        }
        if getattr(motion_cfg, "response_length", None) is not None:
            sampling_params["max_tokens"] = motion_cfg.response_length
        if getattr(motion_cfg, "calculate_log_probs", False):
            sampling_params["logprobs"] = True

        new_non_tensor = copy.deepcopy(batch.non_tensor_batch)
        full_prompts = new_non_tensor.get("full_prompts")
        raw_prompt = new_non_tensor.get("raw_prompt")
        raw_prompt_list = None
        if raw_prompt is not None:
            raw_prompt_list = raw_prompt.tolist() if hasattr(raw_prompt, "tolist") else list(raw_prompt)
            for i, msgs in enumerate(raw_prompt_list):
                if isinstance(msgs, list) and msgs and isinstance(msgs[0], list):
                    raise RuntimeError(
                        f"raw_prompt[{i}] is nested (list of lists); expected list of dicts. "
                        "Check upstream raw_prompt construction."
                    )
            if prefill is not None and prefill in motion_prompts:
                prefill_msg_list = []
                for i, msgs in enumerate(raw_prompt_list):
                    prefill_msg = copy.deepcopy(motion_prompts[prefill])
                    prompt_utils_ppo.format_prompt(
                        new_non_tensor, i, prefill_msg,
                        prev_message=msgs[-1]["content"] if msgs else None,
                    )
                    prefill_msg_list.append(prefill_msg)
                raw_prompt_list = [msgs + [copy.deepcopy(pfm)] for msgs, pfm in zip(raw_prompt_list, prefill_msg_list)]
            vllm_prompts = [
                prompt_utils_ppo.prepare_inputs_for_vllm(messages, motion_processor)
                for messages in raw_prompt_list
            ]
            full_prompts = np.array([p["prompt"] for p in vllm_prompts], dtype=object)
            new_non_tensor["raw_prompt"] = raw_prompt_list
            new_non_tensor["full_prompts"] = [p["prompt"] for p in vllm_prompts]
            new_non_tensor["multi_modal_data"] = [p["multi_modal_data"] for p in vllm_prompts]
        elif full_prompts is None:
            raise RuntimeError("motion generation requires full_prompts or raw_prompt in non_tensor_batch")

        multi_modal_data = new_non_tensor.get("multi_modal_data")
        uid = new_non_tensor.get("uid")
        tasks = []
        full_prompts_list = full_prompts.tolist() if hasattr(full_prompts, "tolist") else full_prompts
        prompt_ids_list = motion_tokenizer.batch_encode_plus(
            full_prompts_list,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        prompt_ids_batch = prompt_ids_list["input_ids"]
        for i in range(len(batch)):
            prompt_ids = prompt_ids_batch[i]
            image_data = None
            video_data = None
            if multi_modal_data is not None:
                item = multi_modal_data[i]
                if isinstance(item, dict):
                    image_data = item.get("image")
                    video_data = item.get("video")
            request_id = uid[i] if uid is not None else uuid4().hex
            tasks.append(
                asyncio.create_task(
                    motion_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=dict(sampling_params),
                        image_data=image_data,
                        video_data=video_data,
                    )
                )
            )

        outputs = await asyncio.gather(*tasks)
        responses = motion_tokenizer.batch_decode(
            [out.token_ids for out in outputs],
            skip_special_tokens=True,
        )

        if new_non_tensor.get("motion_response", None) is None:
            new_non_tensor["motion_response"] = np.array([[r] for r in responses], dtype=object)
        else:
            old = new_non_tensor["motion_response"].tolist()
            new_non_tensor["motion_response"] = np.array(
                [prev + [resp] for prev, resp in zip(old, responses)], dtype=object
            )

        if raw_prompt_list is None:
            raise RuntimeError("raw_prompt is required to update prompts after motion generation")
        if isinstance(new_non_tensor.get("raw_prompt"), np.ndarray):
            new_non_tensor["raw_prompt"] = new_non_tensor["raw_prompt"].tolist()
        if isinstance(new_non_tensor.get("full_prompts"), np.ndarray):
            new_non_tensor["full_prompts"] = new_non_tensor["full_prompts"].tolist()
        if isinstance(new_non_tensor.get("multi_modal_data"), np.ndarray):
            new_non_tensor["multi_modal_data"] = new_non_tensor["multi_modal_data"].tolist()

        ann_payload = self.draw_path(new_non_tensor, cur_idx=-1)
        for i in range(len(responses)):
            ann_img = ann_payload["image"][i]
            prompt_utils_ppo.update_prompt_after_response(
                new_non_tensor,
                i,
                responses[i],
                motion_processor,
                motion_prompts,
                prefill=prefill,
                postfill=postfill,
                ann=ann_img,
                format_prompt_fn=prompt_utils_ppo.format_prompt,
            )
        new_non_tensor["full_prompts"] = np.array(new_non_tensor["full_prompts"], dtype=object)
        new_non_tensor["raw_prompt"] = np.array(new_non_tensor["raw_prompt"], dtype=object)
        if new_non_tensor.get("multi_modal_data") is not None:
            new_non_tensor["multi_modal_data"] = np.array(new_non_tensor["multi_modal_data"], dtype=object)

        return DataProto(batch=batch.batch, non_tensor_batch=new_non_tensor, meta_info=batch.meta_info)

    @tqbridge()
    async def generate_judge_sequences(self, batch: DataProto) -> DataProto:
        """Generate judge/critic text using the critic extra rollout. Updates raw_prompt, full_prompts, and multi_modal_data the same way as generate_motion_sequences (via update_prompt_after_response) so the next rollout step sees the critic response and optional postfill in the conversation."""
        if "critic" not in self.extra_managers:
            raise RuntimeError(
                "critic rollout is not initialized; add a 'critic' entry to config.actor_rollout_ref.extra_rollouts"
            )
        critic_manager = self.extra_managers["critic"]
        critic_tokenizer = self.extra_tokenizers["critic"]
        critic_processor = self.extra_processors.get("critic")
        critic_prompts = self.extra_prompts.get("critic", {})
        full_prompts = batch.non_tensor_batch.get("full_prompts")
        if full_prompts is None:
            raise RuntimeError("generate_judge_sequences requires full_prompts in non_tensor_batch")
        full_prompts_list = full_prompts.tolist() if hasattr(full_prompts, "tolist") else full_prompts
        multi_modal_data = batch.non_tensor_batch.get("multi_modal_data")
        uid = batch.non_tensor_batch.get("uid")
        prefill = batch.meta_info.get("prefill", None)
        postfill = batch.meta_info.get("postfill", None)
        extra_list = OmegaConf.select(self.config.actor_rollout_ref, "extra_rollouts") or []
        rollout_cfg = next(
            (e.rollout for e in extra_list if e.name == "critic"),
            self.config.actor_rollout_ref.rollout,
        )
        sampling_params = {
            "temperature": getattr(rollout_cfg, "temperature", 0.0),
            "top_p": getattr(rollout_cfg, "top_p", 1.0),
            "repetition_penalty": getattr(rollout_cfg, "repetition_penalty", 1.0),
        }
        if getattr(rollout_cfg, "response_length", None) is not None:
            sampling_params["max_tokens"] = rollout_cfg.response_length
        prompt_ids_list = critic_tokenizer.batch_encode_plus(
            full_prompts_list,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        prompt_ids_batch = prompt_ids_list["input_ids"]
        tasks = []
        for i in range(len(batch)):
            image_data = None
            video_data = None
            if multi_modal_data is not None:
                item = multi_modal_data[i].item() if hasattr(multi_modal_data[i], "item") else multi_modal_data[i]
                if isinstance(item, dict):
                    image_data = item.get("image")
                    video_data = item.get("video")
            request_id = uid[i] if uid is not None else uuid4().hex
            tasks.append(
                asyncio.create_task(
                    critic_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids_batch[i],
                        sampling_params=dict(sampling_params),
                        image_data=image_data,
                        video_data=video_data,
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)
        responses = critic_tokenizer.batch_decode(
            [out.token_ids for out in outputs],
            skip_special_tokens=True,
        )
        new_non_tensor = copy.deepcopy(batch.non_tensor_batch)
        new_non_tensor["critic_response"] = np.array(responses, dtype=object)

        if critic_processor is not None and "raw_prompt" in new_non_tensor:
            raw_prompt_val = new_non_tensor.get("raw_prompt")
            if isinstance(raw_prompt_val, np.ndarray):
                new_non_tensor["raw_prompt"] = raw_prompt_val.tolist()
            full_prompts_val = new_non_tensor.get("full_prompts")
            if isinstance(full_prompts_val, np.ndarray):
                new_non_tensor["full_prompts"] = full_prompts_val.tolist()
            multi_modal_val = new_non_tensor.get("multi_modal_data")
            if isinstance(multi_modal_val, np.ndarray):
                new_non_tensor["multi_modal_data"] = multi_modal_val.tolist()
            for i in range(len(responses)):
                prompt_utils_ppo.update_prompt_after_response(
                    new_non_tensor,
                    i,
                    responses[i],
                    critic_processor,
                    critic_prompts,
                    prefill=prefill,
                    postfill=postfill,
                    ann=None,
                    format_prompt_fn=prompt_utils_ppo.format_prompt,
                )
            new_non_tensor["full_prompts"] = np.array(new_non_tensor["full_prompts"], dtype=object)
            new_non_tensor["raw_prompt"] = np.array(new_non_tensor["raw_prompt"], dtype=object)
            if new_non_tensor.get("multi_modal_data") is not None:
                new_non_tensor["multi_modal_data"] = np.array(new_non_tensor["multi_modal_data"], dtype=object)

        return DataProto(batch=batch.batch, non_tensor_batch=new_non_tensor, meta_info=batch.meta_info)

    @tqbridge()
    async def generate_actor_rollouts_sync(self, chunk: DataProto) -> DataProto:
        """Generate rollouts from the primary actor only (no agent loop, no tools). Returns DataProto with prompts, responses, input_ids, attention_mask, position_ids, response_mask."""
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        full_prompts = chunk.non_tensor_batch.get("full_prompts")
        if full_prompts is None:
            raise RuntimeError("generate_actor_rollouts_sync requires full_prompts in non_tensor_batch")
        full_prompts_list = full_prompts.tolist() if hasattr(full_prompts, "tolist") else full_prompts
        multi_modal_data = chunk.non_tensor_batch.get("multi_modal_data")
        uid = chunk.non_tensor_batch.get("uid")
        prompt_ids_list = self.tokenizer.batch_encode_plus(
            full_prompts_list,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        prompt_ids_batch = prompt_ids_list["input_ids"]
        tasks = []
        for i in range(len(chunk)):
            image_data = None
            video_data = None
            if multi_modal_data is not None:
                item = multi_modal_data[i]
                if isinstance(item, dict):
                    image_data = item.get("image")
                    video_data = item.get("video")
            request_id = uid[i] if uid is not None else uuid4().hex
            tasks.append(
                asyncio.create_task(
                    self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids_batch[i],
                        sampling_params=dict(sampling_params),
                        image_data=image_data,
                        video_data=video_data,
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)
        response_ids_list = [out.token_ids for out in outputs]
        responses_decoded = self.tokenizer.batch_decode(response_ids_list, skip_special_tokens=True)
        prompt_length = config.prompt_length
        response_length = config.response_length
        self.tokenizer.padding_side = "left"
        prompt_tensors = []
        response_tensors = []
        response_mask_tensors = []
        attention_mask_tensors = []
        for prompt_ids, response_ids in zip(prompt_ids_batch, response_ids_list):
            prompt_out = self.tokenizer.pad(
                {"input_ids": prompt_ids},
                padding="max_length",
                max_length=prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_out["input_ids"].dim() == 1:
                prompt_out["input_ids"] = prompt_out["input_ids"].unsqueeze(0)
                prompt_out["attention_mask"] = prompt_out["attention_mask"].unsqueeze(0)
            self.tokenizer.padding_side = "right"
            resp_out = self.tokenizer.pad(
                {"input_ids": response_ids},
                padding="max_length",
                max_length=response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if resp_out["input_ids"].dim() == 1:
                resp_out["input_ids"] = resp_out["input_ids"].unsqueeze(0)
                resp_out["attention_mask"] = resp_out["attention_mask"].unsqueeze(0)
            resp_mask = [1] * len(response_ids) + [0] * (response_length - len(response_ids))
            resp_mask_t = torch.tensor(resp_mask, dtype=torch.long).unsqueeze(0) * resp_out["attention_mask"]
            prompt_tensors.append(prompt_out["input_ids"])
            response_tensors.append(resp_out["input_ids"])
            response_mask_tensors.append(resp_mask_t)
            attention_mask_tensors.append(
                torch.cat([prompt_out["attention_mask"], resp_out["attention_mask"]], dim=1)
            )
            self.tokenizer.padding_side = "left"
        prompts_padded = torch.cat(prompt_tensors, dim=0)
        responses_padded = torch.cat(response_tensors, dim=0)
        response_mask_padded = torch.cat(response_mask_tensors, dim=0)
        attention_mask = torch.cat(attention_mask_tensors, dim=0)
        input_ids = torch.cat([prompts_padded, responses_padded], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {
            "prompts": prompts_padded,
            "responses": responses_padded,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask_padded,
        }
        new_non_tensor = copy.deepcopy(chunk.non_tensor_batch)
        new_non_tensor["responses_decoded"] = np.array(responses_decoded, dtype=object)
        return DataProto(batch=batch_dict, non_tensor_batch=new_non_tensor, meta_info=chunk.meta_info)

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                dataset_config=self.config.data,
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, **kwargs)

    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        self.tokenizer.padding_side = "left"
        assert len(output.prompt_ids) <= self.config.actor_rollout_ref.rollout.prompt_length, (
            f"Prompt length {len(output.prompt_ids)} exceeds max prompt length "
            f"{self.config.actor_rollout_ref.rollout.prompt_length}"
        )
        assert len(output.response_ids) <= self.config.actor_rollout_ref.rollout.response_length, (
            f"Response length {len(output.response_ids)} exceeds max response length "
            f"{self.config.actor_rollout_ref.rollout.response_length}"
        )
        assert len(output.response_mask) <= self.config.actor_rollout_ref.rollout.response_length, (
            f"Response mask length {len(output.response_mask)} exceeds max response length "
            f"{self.config.actor_rollout_ref.rollout.response_length}"
        )

        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                experts_tensor = torch.from_numpy(output.routed_experts)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)
        # breakpoint()
        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(input_ids, attention_mask, multi_modal_inputs)
        await self._compute_score(
            output,
            prompts=prompt_output["input_ids"],
            responses=response_output["input_ids"],
            attention_mask=attention_mask,
            input_ids=input_ids,
            position_ids=position_ids,
            kwargs=kwargs,
        )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image and video."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        images = output.multi_modal_data.get("image")
        videos = output.multi_modal_data.get("video")
        # split the videos and according metadatas
        if videos is not None:
            videos, video_metadatas = zip(*videos, strict=False)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        multi_modal_inputs = self.processor(
            text=[current_text],
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        video_grid_thw = multi_modal_inputs.get("video_grid_thw")
        if video_grid_thw is not None:
            videos_seqlens = torch.repeat_interleave(video_grid_thw[:, 1] * video_grid_thw[:, 2], video_grid_thw[:, 0])
            multi_modal_inputs["videos_seqlens"] = videos_seqlens
        return multi_modal_inputs

    def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        video_grid_thw = multi_modal_inputs.get("video_grid_thw")

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, output, prompts, responses, attention_mask, input_ids, position_ids, kwargs):
        """Compute reward score for single sample."""
        enable_async_reward = (
            self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
        ) or not self.config.reward_model.enable
        defer_reward = bool(kwargs.get("__defer_reward__", False))
        
        if output.reward_score is None and enable_async_reward and self.use_reward_loop and not defer_reward:
            batch = TensorDict(
                {
                    "prompts": prompts,  # [1, prompt_length]
                    "responses": responses,  # [1, response_length]
                    "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                    "input_ids": input_ids,  # [1, prompt_length + response_length]
                    "position_ids": position_ids,
                },
                batch_size=1,
            )
            non_tensor_batch = {
                **{k: np.array([v]) for k, v in kwargs.items()},
                "__num_turns__": np.array([output.num_turns]),
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
            }

            data = DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
            )
            result = await self.reward_loop_worker.compute_score.remote(data)
            output.reward_score = result["reward_score"]
            output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        # Also preserve raw multi_modal_data (PIL images / video tuples) so downstream
        # consumers (e.g. build_rm_raw_prompt) can rebuild chat payloads with real media.
        multi_modal_data_list = [input.multi_modal_data for input in inputs]
        if any(mmd is not None for mmd in multi_modal_data_list):
            non_tensor_batch["multi_modal_data"] = np.array(multi_modal_data_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

    def create_transferqueue_client(
        self,
    ):
        """Create a client for data system (TransferQueue)."""
        from verl.single_controller.ray.base import get_random_string
        from verl.utils.transferqueue_utils import create_transferqueue_client

        client_name = get_random_string(length=6)

        self.tq_client = create_transferqueue_client(
            client_id=f"AgentLoopWorker_{client_name}",
            config=self.config.transfer_queue,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rm_resource_pool: RayResourcePool = None,
        rollout_resource_pool: RayResourcePool = None,
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
            rm_resource_pool (RayResourcePool): Resource pool for reward model (Standalone mode).
        """
        self.config = config
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward_loop import RewardModelManager

            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

        self._initialize_llm_servers()
        self.extra_rollout_replicas = {}
        self.extra_server_handles = {}
        self.extra_server_addresses = {}
        self.extra_rollout_wake_up = {}
        self.extra_rollout_sleep = {}
        self._extra_rollout_entries = {}

        # Release actor rollout weights before initializing extra rollout servers.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        self._initialize_extra_rollout_servers()
        self._init_agent_loop_workers()

        # Initially put extra rollouts that use free_cache_engine to sleep.
        for name, replicas in self.extra_rollout_replicas.items():
            entry = self._extra_rollout_entries.get(name)
            if entry and getattr(entry.rollout, "free_cache_engine", True) and replicas:
                self.extra_rollout_sleep[name]()

    def _initialize_llm_servers(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        if world_size % rollout_world_size != 0:
            raise ValueError(
                "rollout world_size mismatch: "
                f"world_size={world_size} is not divisible by rollout_world_size={rollout_world_size}"
            )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group:
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: {self.server_addresses}")
        if self.worker_group:
            try:
                ready_flags = ray.get(self.worker_group.execute_all("rollout_engine_ready"))
                if not all(ready_flags):
                    not_ready = [idx for idx, ready in enumerate(ready_flags) if not ready]
                    logger.warning(f"rollout_engine_ready is False for ranks: {not_ready}")
            except Exception as exc:
                logger.warning(f"rollout_engine_ready check failed: {exc}")

        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(rollout_config.prometheus, self.server_addresses, rollout_config.name)

    def _initialize_extra_rollout_servers(self):
        extra_rollouts_list = OmegaConf.select(self.config.actor_rollout_ref, "extra_rollouts")
        if extra_rollouts_list is None or (hasattr(extra_rollouts_list, "__len__") and len(extra_rollouts_list) == 0):
            return

        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )

        for entry in extra_rollouts_list:
            name = entry.name
            rollout_cfg = entry.rollout
            model_cfg = entry.model
            sync_with_actor = entry.get("sync_with_actor", False)

            # vLLM sleep mode allows only one instance per process. When motion (or any extra)
            # syncs with the actor, use the primary rollout instead of creating a second hybrid.
            if sync_with_actor and self.worker_group:
                self.extra_rollout_replicas[name] = []
                self.extra_server_handles[name] = list(self.server_handles)
                self.extra_server_addresses[name] = list(self.server_addresses)
                self._extra_rollout_entries[name] = entry
                self.extra_rollout_wake_up[name] = self.wake_up
                self.extra_rollout_sleep[name] = self.sleep
                logger.info(
                    "AgentLoopManager: extra rollout '%s' uses primary rollout (sync_with_actor=true, one vLLM per process).",
                    name,
                )
                continue

            rollout_world_size = (
                rollout_cfg.tensor_model_parallel_size
                * rollout_cfg.data_parallel_size
                * rollout_cfg.pipeline_model_parallel_size
            )
            if world_size % rollout_world_size != 0:
                raise ValueError(
                    f"extra rollout '{name}' world_size mismatch: "
                    f"world_size={world_size} is not divisible by rollout_world_size={rollout_world_size}"
                )
            num_replicas = world_size // rollout_world_size

            replica_class = get_rollout_replica_class(rollout_cfg.name)
            replicas = [
                replica_class(
                    replica_rank=replica_rank,
                    config=rollout_cfg,
                    model_config=model_cfg,
                    gpus_per_node=self.config.trainer.n_gpus_per_node,
                )
                for replica_rank in range(num_replicas)
            ]

            if self.rollout_resource_pool is not None:
                logger.info("AgentLoopManager: initializing extra rollout '%s' in colocated mode.", name)
                if num_replicas > 1:
                    # Multiple replicas share the same pool; each must use only its slice of workers
                    # so that len(workers) == world_size (tp * dp * pp) per replica.
                    shared_wg = RayWorkerGroup(
                        resource_pool=self.rollout_resource_pool,
                        ray_cls_with_init=replicas[0].get_ray_class_with_init_args(),
                        bin_pack=False,
                        name_prefix=f"rollout_colocate_extra_{name}",
                    )
                    self._run_all(
                        [
                            server.init_colocated(worker_group=shared_wg, replica_rank=i)
                            for i, server in enumerate(replicas)
                        ]
                    )
                else:
                    self._run_all([server.init_colocated(self.rollout_resource_pool) for server in replicas])
            else:
                self._run_all([server.init_standalone() for server in replicas])

            self.extra_rollout_replicas[name] = replicas
            self.extra_server_handles[name] = [s._server_handle for s in replicas]
            self.extra_server_addresses[name] = [s._server_address for s in replicas]
            self._extra_rollout_entries[name] = entry

            def _make_wake_sleep(replica_list):
                def wake():
                    self._run_all([r.wake_up() for r in replica_list])

                def sleep():
                    self._run_all([r.sleep() for r in replica_list])

                return wake, sleep

            wake_fn, sleep_fn = _make_wake_sleep(replicas)
            self.extra_rollout_wake_up[name] = wake_fn
            self.extra_rollout_sleep[name] = sleep_fn

            print(f"AgentLoopManager (extra rollout '{name}'): {self.extra_server_addresses[name]}")
            if getattr(rollout_cfg, "prometheus", None) and getattr(rollout_cfg.prometheus, "enable", False):
                if getattr(rollout_cfg, "disable_log_stats", True):
                    raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
                update_prometheus_config(
                    rollout_cfg.prometheus, self.extra_server_addresses[name], f"{rollout_cfg.name}_{name}"
                )

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        extra_rollouts_list = OmegaConf.select(self.config.actor_rollout_ref, "extra_rollouts")

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.server_handles,
                    self.reward_router_address,
                    self.extra_server_handles,
                    extra_rollouts_list,
                )
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        # Fix for Issue #4147: Always call wake_up() to ensure weight sync
        # The wake_up()/sleep() methods internally check fzree_cache_engine
        self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        # breakpoint()
        output = DataProto.concat(outputs)
        # Fix for Issue #4147: Always call sleep() to ensure proper cleanup
        self.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def generate_motion_sequences(self, prompts: DataProto) -> DataProto:
        """Generate motion sequences via extra_rollouts['motion'] (primary when sync_with_actor, else dedicated)."""
        if "motion" not in self.extra_rollout_wake_up or not self.extra_server_handles.get("motion"):
            raise RuntimeError(
                "motion rollout is not initialized; add a 'motion' entry to config.actor_rollout_ref.extra_rollouts"
            )
        self.extra_rollout_wake_up["motion"]()
        chunks = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_motion_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunks, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        self.extra_rollout_sleep["motion"]()
        return output

    def generate_judge_sequences(self, prompts: DataProto) -> DataProto:
        """Generate judge/critic text via the critic extra rollout servers."""
        if "critic" not in self.extra_rollout_wake_up:
            raise RuntimeError(
                "critic rollout is not initialized; add a 'critic' entry to config.actor_rollout_ref.extra_rollouts"
            )
        self.extra_rollout_wake_up["critic"]()
        chunks = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_judge_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunks, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        self.extra_rollout_sleep["critic"]()
        return output

    def generate_extra_rollout_sequences(self, rollout_name: str, prompts: DataProto) -> DataProto:
        """Generate sequences using a named extra rollout (dispatches to motion or critic)."""
        if rollout_name == "motion":
            return self.generate_motion_sequences(prompts)
        if rollout_name == "critic":
            return self.generate_judge_sequences(prompts)
        raise ValueError(
            f"Unknown extra rollout name '{rollout_name}'; "
            "supported: 'motion', 'critic'. Add more in AgentLoopManager.generate_extra_rollout_sequences if needed."
        )

    def generate_actor_rollouts(self, prompts: DataProto) -> DataProto:
        """Generate rollouts from the primary actor only (no agent loop, no tools)."""
        self.wake_up()
        chunks = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_actor_rollouts_sync.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunks, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        self.sleep()
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def clear_kv_cache(self):
        """Clear all rollout kv cache, but don`t sleep."""
        self._run_all([replica.clear_kv_cache() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
