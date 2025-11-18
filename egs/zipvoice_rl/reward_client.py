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
"""
Reward calculation client for ASR reward server.
"""

from __future__ import annotations

import base64
import warnings
from typing import List

import numpy as np
import requests
import torch

REWARD_SERVER_URL = "http://localhost:8000/v2/models/asr_reward/infer"


def asr_reward_computation(
    wavs: List[torch.Tensor], texts: List[str], timeout: float = 200.0
) -> (dict, dict):
    """
    Send wavs and ground-truth texts to the Triton server and get reward.
    wavs: list of 1-D float tensors (or numpy arrays)
    texts: list of strings
    """
    if not wavs:
        return {"asr_reward": []}, {"transcripts": []}

    # 1. Pad wavs to the same length
    wavs = [w.squeeze(0) for w in wavs]
    wav_lens = np.array([[len(w)] for w in wavs], dtype=np.int32)
    max_len = np.max(wav_lens)

    padded_wavs = []
    for w in wavs:
        pad_width = max_len - len(w)
        # using numpy for padding
        if isinstance(w, torch.Tensor):
            w = w.cpu().numpy()
        padded_w = np.pad(w, (0, pad_width), mode="constant", constant_values=0)
        padded_wavs.append(padded_w)

    wavs_arr = np.array(padded_wavs, dtype=np.float32)
    # 2. Prepare payload
    payload = {
        "inputs": [
            {
                "name": "WAV",
                "shape": list(wavs_arr.shape),
                "datatype": "FP32",
                "data": wavs_arr.tolist(),
            },
            {
                "name": "WAV_LENS",
                "shape": list(wav_lens.shape),
                "datatype": "INT32",
                "data": wav_lens.tolist(),
            },
            {
                "name": "GT_TEXT",
                "shape": [len(texts), 1],
                "datatype": "BYTES",
                "data": texts,
            },
        ]
    }

    # 3. Send request and get response
    try:
        rsp = requests.post(
            REWARD_SERVER_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
            verify=False,
            params={"request_id": "0"},
        )
        rsp.raise_for_status()
        result = rsp.json()
    except Exception as e:
        warnings.warn(f"Remote reward server error: {e}; returning 0.0 rewards")
        rewards = [0.0] * len(wavs)
        transcripts = [""] * len(wavs)
        return {"asr_reward": rewards}, {"transcripts": transcripts}

    # 4. Parse response
    try:
        # Reward is returned as the first output
        rewards = [float(item) for item in result["outputs"][0]["data"]]
        # Transcript is the second output
        transcripts = result["outputs"][1]["data"]

    except (KeyError, IndexError, TypeError) as e:
        warnings.warn(f"Failed to parse reward server response: {e}; returning 0.0 rewards")
        rewards = [0.0] * len(wavs)
        transcripts = [""] * len(wavs)

    return {"asr_reward": rewards}, {"transcripts": transcripts}


if __name__ == "__main__":
    from datasets import load_dataset
    import torchaudio

    print("Testing reward_client.py")

    # Load dataset
    try:
        dataset = load_dataset("yuekai/aishell", "test", trust_remote_code=True)["test"]
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        exit()

    # Get a few samples
    num_samples = 4
    samples = [dataset[i] for i in range(num_samples)]

    wavs = []
    texts = []
    for sample in samples:
        audio_data = sample["audio"]
        audio_array = torch.from_numpy(audio_data["array"]).float()
        sample_rate = audio_data["sampling_rate"]

        # Resample if necessary to 16kHz for ASR
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            audio_array = resampler(audio_array)

        wavs.append(audio_array)
        texts.append(sample["text"])

    print(f"\nPrepared {len(wavs)} samples for testing.")
    print("GT Texts:", texts)

    # Call reward function
    rewards, metadata = asr_reward_computation(wavs, texts)

    print("\n--- Results ---")
    print("Rewards:", rewards)
    print("Transcripts:", metadata.get("transcripts", []))

    # Test with empty input
    print("\n--- Testing with empty input ---")
    rewards, metadata = asr_reward_computation([], [])
    print("Rewards:", rewards)
    print("Metadata:", metadata)
