#!/usr/bin/env python3
# Copyright    2024-2025  Xiaomi Corp.        (authors: Wei Kang,
#                                                       Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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
This script trains a ZipVoice model with GRPO (Generative Retraining with Policy Optimization).
It is adapted from train_tts.py and uses a similar RL-based training loop.

Usage:

python3 -m zipvoice.bin.train_zipvoice_grpo \
    --world-size 8 \
    --use-fp16 1 \
    --exp-dir exp/zipvoice_grpo \
    --pretrained-model zipvoice_distill \
    --dataset-path aishell-3-cosy.jsonl
"""

import argparse
import copy
import json
import logging
import os
import random
import time
from concurrent import futures
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchaudio
import wandb
from datasets import load_dataset
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from pipeline_zipvoice import ZipVoicePipeline
from zipvoice.models.modules.solver import sde_step_with_logprob
from zipvoice.utils.checkpoint import (
    load_checkpoint,
    remove_checkpoints,
    resume_checkpoint,
    save_checkpoint,
    save_checkpoint_with_global_batch_idx,
)
from zipvoice.utils.common import (
    AttributeDict,
    GradScaler,
    cleanup_dist,
    create_grad_scaler,
    get_env_info,
    setup_dist,
    setup_logger,
    str2bool,
    torch_autocast,
)

tqdm = partial(tqdm, dynamic_ncols=True)


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k  # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank  # Current replica rank
        self.seed = seed  # Random seed for synchronization

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()

            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(
                len(repeated_indices), generator=g
            ).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])

            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


class SpeechPromptDataset(Dataset):
    def __init__(
        self,
        target_text_path,
        split="train",
        prompt_dataset_name="yuekai/aishell",
        prompt_dataset_split="test",
    ):
        # Load target texts
        self.target_texts = []
        if target_text_path.endswith(".jsonl"):
            with open(target_text_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.target_texts.append(json.loads(line)["text"])
        elif target_text_path.endswith(".txt"):
            with open(target_text_path, "r", encoding="utf-8") as f:
                self.target_texts = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"Neither {target_text_path} found.")

        # Load prompt dataset
        self.prompt_dataset = load_dataset(
            "yuekai/aishell", "test", trust_remote_code=True
        )["test"]

    def __len__(self):
        return len(self.target_texts)

    def _get_random_prompt(self):
        random_idx = random.randint(0, len(self.prompt_dataset) - 1)
        sample = self.prompt_dataset[random_idx]

        prompt_text = sample["text"].replace(" ", "")

        audio_data = sample["audio"]
        # The audio array is already a numpy array, convert it to a tensor.
        audio_array = torch.from_numpy(audio_data["array"]).float()
        sample_rate = audio_data["sampling_rate"]

        return prompt_text, audio_array, sample_rate

    def __getitem__(self, idx):
        target_text = self.target_texts[idx]
        prompt_text, prompt_wav, prompt_sampling_rate = self._get_random_prompt()

        item = {
            "id": f"item_{idx}",
            "prompt_wav": prompt_wav,
            "prompt_sampling_rate": prompt_sampling_rate,
            "prompt_text": prompt_text,
            "target_text": target_text,
        }
        return item

    @staticmethod
    def collate_fn(batch):
        target_sampling_rate = 24000
        ids, prompt_wavs_list, prompt_texts_list, target_texts_list = [], [], [], []

        for item in batch:
            prompt_wav = item["prompt_wav"]
            prompt_sampling_rate = item["prompt_sampling_rate"]

            if prompt_sampling_rate != target_sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=prompt_sampling_rate,
                    new_freq=target_sampling_rate,
                )
                prompt_wav = resampler(prompt_wav)

            prompt_wavs_list.append(prompt_wav)
            prompt_texts_list.append(item["prompt_text"])
            target_texts_list.append(item["target_text"])
            ids.append(item["id"])

        return ids, prompt_wavs_list, prompt_texts_list, target_texts_list


def compute_log_prob(model, pipeline, sample, j, params):
    """
    Computes the log probability of the next latent state given the current latent state.
    """
    latents = sample["latents"][:, j]
    timesteps = sample["timesteps"]
    step_index = (timesteps[0] == timesteps[:, j][0]).nonzero().item()

    current_t = timesteps[:, step_index]
    next_t = timesteps[:, step_index + 1]

    # Predict the velocity
    v = model.forward_fm_decoder(
        t=current_t,
        xt=latents,
        text_condition=sample["text_condition"],
        speech_condition=sample["speech_condition"],
        padding_mask=sample["padding_mask"],
        guidance_scale=params.guidance_scale,
    )

    # Compute the log prob of next_latents given latents under the current model
    _, log_prob, _, _ = sde_step_with_logprob(
        v,
        sigma_prev=next_t,
        sigma=current_t,
        sample=latents,
        prev_sample=sample["next_latents"][:, j],
        noise_level=params.noise_level,
        sde_type="cps",  # Make sure this matches the one used in sampling
    )

    return log_prob


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12356,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch.""",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/zipvoice_grpo",
        help="""The experiment dir.""",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )
    # New arguments from train_tts.py config
    parser.add_argument("--run-name", type=str, default="tts_rl_8gpu")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--save-freq", type=int, default=20)
    parser.add_argument("--num-checkpoint-limit", type=int, default=5)
    parser.add_argument(
        "--mixed-precision", type=str, default="fp16", choices=["fp16", "bf16", "no"]
    )
    parser.add_argument("--dataset-path", type=str, default="aishell-3-cosy.jsonl")
    parser.add_argument("--pretrained-model", type=str, default="zipvoice_distill")
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--num-audio-per-prompt", type=int, default=4)
    parser.add_argument("--num-batches-per-epoch", type=int, default=8)
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-4)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-inner-epochs", type=int, default=1)
    parser.add_argument("--adv-clip-max", type=float, default=5)
    parser.add_argument("--clip-range", type=float, default=1e-4)
    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=True,
        help="Whether to use half precision training.",
    )
    parser.add_argument(
        "--save-every-n",
        type=int,
        default=5000,
        help="Save checkpoint after processing this number of batches",
    )
    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.""",
    )

    return parser


def train(params: AttributeDict, rank: int, world_size: int):
    """The main training loop."""
    device = torch.device("cuda", rank)
    fix_random_seed(params.seed)

    # Setup logging
    if rank == 0:
        wandb.init(project="flow_grpo_tts", name=params.run_name)
    logging.info(f"\n{params}")

    # Load TTS model via pipeline
    pipeline = ZipVoicePipeline(model_name=params.pretrained_model, device=device)
    model = pipeline.model

    # For now, we train the full model.
    model_parameters = list(model.parameters())

    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model_parameters,
        lr=params.learning_rate,
        betas=(params.adam_beta1, params.adam_beta2),
        weight_decay=params.adam_weight_decay,
        eps=params.adam_epsilon,
    )

    # Dataset and Dataloader
    train_dataset = SpeechPromptDataset(target_text_path=params.dataset_path)
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=params.train_batch_size,
        k=params.num_audio_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=params.seed,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=SpeechPromptDataset.collate_fn,
    )

    scaler = create_grad_scaler(enabled=params.use_fp16)

    # Placeholder reward function
    def placeholder_reward_fn(wavs, texts, metadata):
        rewards = {"placeholder_reward": [random.random() for _ in wavs]}
        return rewards, {}

    reward_fn = placeholder_reward_fn

    executor = futures.ThreadPoolExecutor(max_workers=2)
    autocast = partial(torch_autocast, dtype=torch.float16, enabled=params.use_fp16)

    logging.info("***** Running training *****")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)
    num_train_timesteps = params.num_steps

    while True:
        #################### SAMPLING ####################
        if isinstance(model, DDP):
            model.module.eval()
        else:
            model.eval()

        samples = []
        for i in tqdm(
            range(params.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=rank != 0,
            position=0,
        ):
            ids, prompt_wavs_list, prompt_texts_list, target_texts_list = next(
                train_iter
            )

            with torch.no_grad():
                with autocast():
                    audios, latents, log_probs, timesteps = pipeline(
                        prompt_text=prompt_texts_list,
                        prompt_wav=prompt_wavs_list,
                        text=target_texts_list,
                        num_step=num_train_timesteps,
                        guidance_scale=params.guidance_scale,
                        enable_sde=True,
                        sde_noise_level=params.noise_level,
                    )

            latents = torch.stack(latents, dim=1)
            log_probs = torch.stack(log_probs, dim=1)
            timesteps = timesteps.unsqueeze(0).repeat(latents.size(0), 1)

            # Recompute conditions for training
            prepared_inputs = pipeline.prepare_latents(
                prompt_text=prompt_texts_list,
                prompt_wav=prompt_wavs_list,
                text=target_texts_list,
            )
            unwrapped_model = model.module if isinstance(model, DDP) else model
            text_condition, padding_mask = (
                unwrapped_model.forward_text_inference_ratio_duration(
                    tokens=prepared_inputs["tokens"],
                    prompt_tokens=prepared_inputs["prompt_tokens"],
                    prompt_features_lens=prepared_inputs["prompt_features_lens"],
                    speed=1.0,
                )
            )

            num_frames = text_condition.shape[1]
            prompt_features = prepared_inputs["prompt_features"]
            speech_condition = torch.nn.functional.pad(
                prompt_features, (0, 0, 0, num_frames - prompt_features.size(1))
            )

            rewards_future = executor.submit(reward_fn, audios, target_texts_list, {})

            samples.append(
                {
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "timesteps": timesteps,
                    "text_condition": text_condition,
                    "speech_condition": speech_condition,
                    "padding_mask": padding_mask,
                    "rewards": rewards_future,
                }
            )

        for sample in tqdm(samples, desc="Waiting for rewards", disable=rank != 0):
            rewards, _ = sample["rewards"].result()
            sample["rewards"] = {
                k: torch.as_tensor(v, device=device).float()
                for k, v in rewards.items()
            }

        rewards_cat = {
            key: torch.cat([s["rewards"][key] for s in samples])
            for key in samples[0]["rewards"]
        }
        samples_cat = {
            key: torch.cat([s[key] for s in samples])
            for key in samples[0]
            if key != "rewards"
        }
        samples_cat["rewards"] = rewards_cat
        samples = samples_cat

        # Advantage calculation with global stats
        local_rewards = samples["rewards"]["placeholder_reward"]
        if world_size > 1:
            # Gather rewards from all GPUs
            local_size = torch.tensor([local_rewards.size(0)], device=device)
            all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
            torch.distributed.all_gather(all_sizes, local_size)
            max_size = max(s.item() for s in all_sizes)

            padded_rewards = torch.zeros(max_size, device=device)
            padded_rewards[: local_rewards.size(0)] = local_rewards
            
            all_rewards_padded = [torch.zeros_like(padded_rewards) for _ in range(world_size)]
            torch.distributed.all_gather(all_rewards_padded, padded_rewards)

            # Unpad and concatenate
            all_rewards = []
            for i in range(world_size):
                all_rewards.append(all_rewards_padded[i][:all_sizes[i].item()])
            
            all_rewards_tensor = torch.cat(all_rewards)
            mean = all_rewards_tensor.mean()
            std = all_rewards_tensor.std() + 1e-8
        else:
            mean = local_rewards.mean()
            std = local_rewards.std() + 1e-8
        
        advantages = (local_rewards - mean) / std
        samples["advantages"] = advantages.unsqueeze(1).repeat(1, num_train_timesteps)

        del samples["rewards"]

        #################### TRAINING ####################
        for inner_epoch in range(params.num_inner_epochs):
            if isinstance(model, DDP):
                model.module.train()
            else:
                model.train()
            
            optimizer.zero_grad()
            
            for j in tqdm(
                range(num_train_timesteps),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                disable=rank != 0,
            ):
                with autocast():
                    unwrapped_model = model.module if isinstance(model, DDP) else model
                    log_prob = compute_log_prob(
                        unwrapped_model, pipeline, samples, j, params
                    )

                    advantages = torch.clamp(
                        samples["advantages"][:, j],
                        -params.adv_clip_max,
                        params.adv_clip_max,
                    )
                    ratio = torch.exp(log_prob - samples["log_probs"][:, j])

                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - params.clip_range,
                        1.0 + params.clip_range,
                    )
                    loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    # Normalize loss by number of timesteps for accumulation
                    loss = loss / num_train_timesteps

                scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            # Logging
            if rank == 0:
                wandb.log(
                    {
                        "loss": loss.item() * num_train_timesteps, # Log unnormalized loss
                        "epoch": epoch,
                        "inner_epoch": inner_epoch,
                    },
                    step=global_step,
                )
            global_step += 1
        
        # Checkpointing
        if epoch > 0 and epoch % params.save_freq == 0:
            filename = params.exp_dir / f"epoch-{epoch}.pt"
            save_checkpoint(
                filename=filename,
                params=params,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.num_checkpoint_limit,
                rank=rank,
                prefix="epoch-"
            )


        epoch += 1


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = AttributeDict(vars(args))

    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    os.makedirs(f"{params.exp_dir}/log", exist_ok=True)
    setup_logger(f"{params.exp_dir}/log/log-train")

    if rank == 0:
        logging.info("Params: ")
        logging.info(params)

    train(params, rank, world_size)

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
