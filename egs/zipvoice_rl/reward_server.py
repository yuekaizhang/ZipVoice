# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Pytriton server for ASR"""

import argparse
import logging
from typing import List
import numpy as np
import torch
import sys
from jiwer import wer
from pypinyin import lazy_pinyin

from tn.chinese.normalizer import Normalizer as ZhNormalizer

# Chinese text normalizer (cached globally)
zh_tn_model = ZhNormalizer(
    cache_dir="./cache",
    remove_erhua=False,
    remove_interjections=False,
    remove_puncts=True,
    overwrite_cache=True,
)

from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from pytriton.proxy.types import Request

from omnisense.models import OmniSenseVoiceSmall

sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")

logger = logging.getLogger("asr_reward_server")


def get_reward_value(c):
    k_pe = 12
    exponents = 1.5
    pow_exp_val = np.exp(-k_pe * c ** exponents)
    # return 1.0 - np.tanh(3.0 * c)
    return pow_exp_val

class _ASR_Reward_Server:
    """Wraps a single OmniSenseVoiceSmall model instance for Triton."""

    def __init__(self, device_id: int):
        self.asr_model = OmniSenseVoiceSmall("iic/SenseVoiceSmall", quantize=False, device_id=device_id)
        self.device_id = device_id

    @batch
    def __call__(self, WAV: np.ndarray, WAV_LENS: np.ndarray, GT_TEXT: np.ndarray):
        """
        WAV: np.ndarray, WAV_LENS: np.ndarray
        GT_TEXT: np.ndarray
        """
        # Ensure the default CUDA device is set correctly for this invocation
        torch.cuda.set_device(self.device_id)

        if self.device_id == 0:
            print(f"device_id: {self.device_id}, WAV: {WAV.shape}, WAV_LENS: {WAV_LENS.shape}")

        wavs = [WAV[i, :WAV_LENS[i, 0]] for i in range(len(WAV))]

        # Decode ground-truth text strings (BYTES → str)
        if GT_TEXT.ndim == 2:
            gt_texts = [GT_TEXT[i, 0].decode("utf-8") for i in range(len(GT_TEXT))]
        else:
            gt_texts = [GT_TEXT[i].decode("utf-8") for i in range(len(GT_TEXT))]
        
        results = self.asr_model.transcribe_single_batch(
            wavs,
            language="zh",
            textnorm="woitn",
        )
        texts = [result.text for result in results]

        # ---------------- Reward computation ----------------
        rewards = []
        for gt_text, hyp_text in zip(gt_texts, texts):
            gt_norm = zh_tn_model.normalize(gt_text).lower()
            hyp_norm = zh_tn_model.normalize(hyp_text).lower()

            # don't compute the tone error
            gt_pinyin = lazy_pinyin(gt_norm)
            hyp_pinyin = lazy_pinyin(hyp_norm)

            c = float(wer(" ".join(gt_pinyin), " ".join(hyp_pinyin)))
            reward_val = get_reward_value(c)
            reward_val = max(0.0, min(1.0, reward_val))
            rewards.append(reward_val)
            if self.device_id == 0:
                print(f"gt_text: {gt_norm}, hyp_text: {hyp_norm}, wer: {c:.4f}, reward_val: {reward_val:.4f}")

        transcripts = np.char.encode(np.array(texts).reshape(-1, 1), "utf-8")
        rewards_arr = np.array(rewards, dtype=np.float32).reshape(-1, 1)


        return {"REWARDS": rewards_arr, "TRANSCRIPTS": transcripts}


def _infer_function_factory(device_ids: List[int]):
    """Creates a list of inference functions, one for each requested device ID."""
    infer_funcs = []
    for device_id in device_ids:
        infer_funcs.append(_ASR_Reward_Server(device_id=device_id))
    return infer_funcs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=64,
        help="Batch size of request.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--number-of-instances-per-device",
        type=int,
        default=1,
        help="Number of model instances to load.",
        required=False,
    )
    parser.add_argument(
        "--number-of-devices",
        type=int,
        default=8,
        help="Number of devices to use.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    triton_config = TritonConfig(
        http_port=8000,
        grpc_port=8001,
        metrics_port=8002,
    )

    device_ids = [i for i in range(args.number_of_devices)]
    device_ids = device_ids * args.number_of_instances_per_device

    with Triton(config=triton_config) as triton:
        logger.info("Loading SenseVoice model on device ids: %s", device_ids)
        triton.bind(
            model_name="asr_reward",
            infer_func=_infer_function_factory(device_ids),
            inputs=[
                Tensor(name="WAV", dtype=np.float32, shape=(-1,)),
                Tensor(name="WAV_LENS", dtype=np.int32, shape=(-1,)),
                Tensor(name="GT_TEXT", dtype=bytes, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="REWARDS", dtype=np.float32, shape=(-1,)),
                Tensor(name="TRANSCRIPTS", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(
                max_batch_size=args.max_batch_size,
                batcher=DynamicBatcher(max_queue_delay_microseconds=10000),  # 10ms
            ),
            strict=True,
        )
        logger.info("Serving inference")
        triton.serve()


if __name__ == "__main__":
    main()
