# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum

import torch
from tensordict.tensorclass import NonTensorData


class DatasetPadMode(str, Enum):
    """Padding mode for dataset"""

    RIGHT = "right"
    LEFT_RIGHT = "left_right"
    NO_PADDING = "no_padding"


class SFTTensorCollator:
    """
    A custom collate_fn that handles batching of sequences.
    1. for variable-length sequences, convert them into NestedTensors.
    2. for fixed-length sequences, use default_collate.
    """

    def __init__(self, pad_mode: DatasetPadMode = DatasetPadMode.LEFT_RIGHT):
        self.pad_mode = pad_mode

    def __call__(self, batch: list[dict[str, any]]) -> dict[str, any]:
        if self.pad_mode == DatasetPadMode.NO_PADDING:
            return self.collate_variable_batch(batch)
        elif self.pad_mode in [DatasetPadMode.RIGHT, DatasetPadMode.LEFT_RIGHT]:
            from torch.utils.data import default_collate

            return default_collate(batch)
        else:
            raise NotImplementedError(f"pad_mode {self.pad_mode} not implemented")

    def collate_variable_batch(self, batch: list[dict[str, any]]) -> dict[str, any]:
        """
        Collates a list of samples into a single batch.

        Args:
            batch: A list of dictionary samples from the dataset.

        Returns:
            A dictionary representing the batched data, with variable-length
            sequences converted to NestedTensors.
        """

        final_batch = {}

        tensor_keys = set().union(*(d.keys() for d in batch))

        # Handle tensor values by creating a NestedTensor.
        for key in tensor_keys:
            if isinstance(batch[0][key], torch.Tensor):
                tensors = [item[key] for item in batch]
                final_batch[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
            else:
                tensors = [NonTensorData(item.get(key)) for item in batch]
                final_batch[key] = torch.stack(tensors, dim=0)

        return final_batch


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class RMTensorCollator:
    """Collates RMDataset pairs into an interleaved flat batch for the engine.

    Each item from RMDataset has:
      - ``input_ids``:      (2, seq_len) — index 0 = chosen, 1 = rejected
      - ``attention_mask``: (2, seq_len)
      - ``position_ids``:   (2, seq_len) or (2, 4, seq_len) for Qwen3-VL
      - ``chosen_multi_modal_inputs``  (optional): dict[str, Tensor]
      - ``rejected_multi_modal_inputs`` (optional): dict[str, Tensor]

    Output is a dict with nested tensors of shape (2*N, *) in interleaved order:
      [chosen_0, rejected_0, chosen_1, rejected_1, ...]
    plus a synthetic ``loss_mask`` equal to ``attention_mask`` (required by the
    engine's ``forward_backward_batch`` to compute ``batch_num_tokens``).
    """

    def __call__(self, batch: list[dict]) -> dict:
        N = len(batch)
        all_input_ids: list[torch.Tensor] = []
        all_attention_mask: list[torch.Tensor] = []
        all_position_ids: list[torch.Tensor] = []
        all_mm: list[NonTensorData] = []

        has_mm = "chosen_multi_modal_inputs" in batch[0]

        for item in batch:
            # index 0 = chosen, index 1 = rejected
            all_input_ids.append(item["input_ids"][0])
            all_input_ids.append(item["input_ids"][1])
            all_attention_mask.append(item["attention_mask"][0])
            all_attention_mask.append(item["attention_mask"][1])
            all_position_ids.append(item["position_ids"][0])
            all_position_ids.append(item["position_ids"][1])
            if has_mm:
                all_mm.append(NonTensorData(item.get("chosen_multi_modal_inputs", None)))
                all_mm.append(NonTensorData(item.get("rejected_multi_modal_inputs", None)))

        result = {
            "input_ids": torch.nested.as_nested_tensor(all_input_ids, layout=torch.jagged),
            "attention_mask": torch.nested.as_nested_tensor(all_attention_mask, layout=torch.jagged),
            "position_ids": torch.nested.as_nested_tensor(all_position_ids, layout=torch.jagged),
            # Synthetic loss_mask: engine uses data["loss_mask"].sum() for batch_num_tokens.
            # We use attention_mask (all valid tokens) as a proxy.
            "loss_mask": torch.nested.as_nested_tensor(
                [m.clone() for m in all_attention_mask], layout=torch.jagged
            ),
        }

        if has_mm and all_mm:
            result["multi_modal_inputs"] = torch.stack(all_mm, dim=0)

        return result

