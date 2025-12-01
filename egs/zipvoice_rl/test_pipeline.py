import argparse
import logging
import os
from functools import partial

import torch
import torch.multiprocessing as mp
import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from zipvoice.rl.pipeline_zipvoice import ZipVoicePipeline
from zipvoice.utils.common import cleanup_dist, setup_dist
from reward_client import asr_reward_computation
from scripts.offline_decode_files import (
    store_transcripts,
    write_error_stats,
    normalize_text_alimeeting,
)
from typing import List, Tuple, Dict, Iterable, TextIO, Union
from pathlib import Path
from collections import defaultdict
import kaldialign
import torch.distributed as dist
import numpy as np

tqdm = partial(tqdm, dynamic_ncols=True)
Pathlike = Union[str, Path]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12356,
        help="Master port to use for DDP.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_pipeline",
        help="Directory to save the generated audio files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.7,
        help="SDE noise level for ZipVoice pipeline.",
    )
    parser.add_argument(
        "--huggingface-dataset-split",
        type=str,
        default="test",
        help="The split of huggingface dataset to use for inference.",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=4,
        help="Number of steps for ZipVoice pipeline.",
    )
    parser.add_argument(
        "--rollout-n",
        type=int,
        default=1,
        help="Number of rollouts for ZipVoice pipeline.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Guidance scale for ZipVoice pipeline.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice_distill",
        help="Model name for ZipVoice pipeline.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model directory for ZipVoice pipeline.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="epoch-1.pt",
        help="Checkpoint name for ZipVoice pipeline.",
    )   
    parser.add_argument(
        "--enable-sde",
        action="store_true",
        help="Enable SDE-based sampling.",
    )
    parser.add_argument(
        "--enable-ln-sigma-sampling",
        action="store_true",
        help="Enable sampling with ln_sigma.",
    )
    return parser


def collate_fn(batch):
    target_sampling_rate = 24000
    ids, prompt_wavs_list, prompt_texts_list, target_texts_list = [], [], [], []

    for item in batch:
        prompt_wav = torch.from_numpy(item["prompt_audio"]["array"]).float()
        prompt_sampling_rate = item["prompt_audio"]["sampling_rate"]

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


def run(rank, world_size, args):
    if world_size > 1:
        setup_dist(rank, world_size, args.master_port)

    device = torch.device("cuda", rank)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
    model_dir = args.model_dir
    checkpoint_name = args.checkpoint_name
    if model_dir is not None:
        tokenizer_type = "libritts" if "libritts" in model_dir else "emilia"
        pipeline = ZipVoicePipeline(model_name=args.model_name, model_dir=model_dir, checkpoint_name=checkpoint_name, tokenizer_type=tokenizer_type, device=device)
    else:
        pipeline = ZipVoicePipeline(model_name=args.model_name, device=device)
    dataset_name = "yuekai/CV3-Eval" if 'zero' in args.huggingface_dataset_split else "yuekai/seed_tts_cosy2"
    dataset = load_dataset(
        dataset_name,
        split=args.huggingface_dataset_split,
        trust_remote_code=True,
    )
    # only select the first 20 items
    dataset = dataset.select(range(20))

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=False)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        sampler=sampler,
    )

    # Store results for the current rank
    ids_list, target_texts_list_rank, transcripts_list_rank, rewards_list_rank = (
        [],
        [],
        [],
        [],
    )

    for batch in tqdm(data_loader, disable=rank != 0):
        ids, prompt_wavs_list, prompt_texts_list, target_texts_list = batch

        if args.rollout_n > 1:
            all_rollout_wavs = []
            all_rollout_rewards = []
            all_rollout_transcripts = []

            for rollout_idx in range(args.rollout_n):
                pipeline_output = pipeline(
                    prompt_text=prompt_texts_list,
                    prompt_wav=prompt_wavs_list,
                    text=target_texts_list,
                    num_step=args.num_step,
                    guidance_scale=args.guidance_scale,
                    enable_sde=args.enable_sde,
                    sde_noise_level=args.noise_level,
                    enable_ln_sigma_sampling=args.enable_ln_sigma_sampling,
                )
                # if args.enable_sde:
                output_wavs_rollout, _, _, _ = pipeline_output
                # else:
                    # output_wavs_rollout = pipeline_output

                rewards_rollout, metadata = asr_reward_computation(
                    output_wavs_rollout, target_texts_list
                )
                transcripts_rollout = metadata.get(
                    "transcripts", [""] * len(output_wavs_rollout)
                )

                all_rollout_wavs.append(output_wavs_rollout)
                all_rollout_rewards.append(rewards_rollout["asr_reward"])
                all_rollout_transcripts.append(transcripts_rollout)

            best_rewards_batch = []
            best_transcripts_batch = []

            # For each item in the batch
            for i in range(len(ids)):
                # Find the best rollout for this item
                best_rollout_idx = np.argmax(
                    [all_rollout_rewards[r][i] for r in range(args.rollout_n)]
                )

                # Save all wavs, renaming the best one
                for r_idx in range(args.rollout_n):
                    wav_to_save = all_rollout_wavs[r_idx][i]
                    if r_idx == best_rollout_idx:
                        save_path = f"{args.results_dir}/{ids[i]}.wav"
                    else:
                        save_path = f"{args.results_dir}/{ids[i]}_{r_idx}.wav"
                    torchaudio.save(
                        save_path,
                        wav_to_save.cpu(),
                        sample_rate=pipeline.sampling_rate,
                    )

                best_rewards_batch.append(all_rollout_rewards[best_rollout_idx][i])
                best_transcripts_batch.append(
                    all_rollout_transcripts[best_rollout_idx][i]
                )

            rewards = {"asr_reward": best_rewards_batch}
            transcripts = best_transcripts_batch

        else:
            pipeline_output = pipeline(
                prompt_text=prompt_texts_list,
                prompt_wav=prompt_wavs_list,
                text=target_texts_list,
                num_step=args.num_step,
                enable_sde=args.enable_sde,
                sde_noise_level=args.noise_level,
                enable_ln_sigma_sampling=args.enable_ln_sigma_sampling,
            )
            if args.enable_sde:
                output_wavs, _, _, _ = pipeline_output
            else:
                output_wavs = pipeline_output

            rewards, metadata = asr_reward_computation(output_wavs, target_texts_list)
            transcripts = metadata.get("transcripts", [""] * len(output_wavs))

            for i, wav in enumerate(output_wavs):
                save_path = f"{args.results_dir}/{ids[i]}.wav"
                torchaudio.save(save_path, wav.cpu(), sample_rate=pipeline.sampling_rate)

        ids_list.extend(ids)
        target_texts_list_rank.extend(target_texts_list)
        transcripts_list_rank.extend(transcripts)
        rewards_list_rank.extend(rewards["asr_reward"])

    if world_size > 1:
        dist.barrier()  # wait for all processes to finish inference

        # gather all results
        gathered_ids = [None] * world_size
        dist.all_gather_object(gathered_ids, ids_list)

        gathered_targets = [None] * world_size
        dist.all_gather_object(gathered_targets, target_texts_list_rank)

        gathered_transcripts = [None] * world_size
        dist.all_gather_object(gathered_transcripts, transcripts_list_rank)

        gathered_rewards = [None] * world_size
        dist.all_gather_object(gathered_rewards, rewards_list_rank)

        if rank == 0:
            # Flatten lists
            all_ids = [item for sublist in gathered_ids for item in sublist]
            all_targets = [item for sublist in gathered_targets for item in sublist]
            all_transcripts = [
                item for sublist in gathered_transcripts for item in sublist
            ]
            all_rewards = [item for sublist in gathered_rewards for item in sublist]
    else:
        all_ids = ids_list
        all_targets = target_texts_list_rank
        all_transcripts = transcripts_list_rank
        all_rewards = rewards_list_rank

    if rank == 0:
        final_results = []
        for i in range(len(all_ids)):
            normalized_target = normalize_text_alimeeting(all_targets[i])
            final_results.append((all_ids[i], normalized_target, all_transcripts[i]))

        store_transcripts(
            filename=f"{args.results_dir}/recogs-sensevoice.txt", texts=final_results
        )
        errs_file = f"{args.results_dir}/errs-sensevoice.txt"
        with open(
            errs_file, "w", encoding="utf-8"
        ) as f:
            write_error_stats(f, "test-set", final_results, enable_log=True)

        with open(errs_file, "r") as f:
            print(f.readline().strip())  # WER
            print(f.readline().strip())  # Detailed errors

        # --- Reward Statistics ---
        rewards_arr = np.array(all_rewards)
        mean_reward = np.mean(rewards_arr)
        std_reward = np.std(rewards_arr)
        variance_reward = np.var(rewards_arr)

        stats_output = []
        stats_output.append("--- Reward Statistics ---")
        stats_output.append(f"Mean reward: {mean_reward:.4f}")
        stats_output.append(f"Standard deviation of reward: {std_reward:.4f}")
        stats_output.append(f"Variance of reward: {variance_reward:.4f}")

        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        reward_quantiles = np.quantile(rewards_arr, quantiles)
        stats_output.append("\nReward distribution (quantiles):")
        for q, val in zip(quantiles, reward_quantiles):
            stats_output.append(f"  {int(q*100)}th percentile: {val:.4f}")

        hist, bin_edges = np.histogram(rewards_arr)
        stats_output.append("\nReward distribution (histogram):")
        for i in range(len(hist)):
            stats_output.append(f"  {bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {hist[i]}")

        # Print to console
        print("\n" + "\n".join(stats_output))

        # Save to file
        with open(f"{args.results_dir}/rewards.txt", "w") as f:
            f.write("\n".join(stats_output))

    if world_size > 1:
        cleanup_dist()


def main():
    parser = get_parser()
    args = parser.parse_args()

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
