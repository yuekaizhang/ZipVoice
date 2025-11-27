#!/bin/bash

# This script is an example of fine-tuning ZipVoice on your custom datasets.

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

export HF_HOME="./hf_cache"

stage=$1
stop_stage=$2


# Whether the language of training data is one of Chinese and English
is_zh_en=1

# Language identifier, used when language is not Chinese or English
# see https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md
# Example of French: lang=fr
lang=default

if [ $is_zh_en -eq 1 ]; then
      tokenizer=emilia
else
      tokenizer=espeak
      [ "$lang" = "default" ] && { echo "Error: lang is not set!" >&2; exit 1; }
fi

# You can set `max_len` according to statistics from the command 
# `lhotse cut describe data/fbank/custom_cuts_train.jsonl.gz`.
# Set `max_len` to 99% duration.

# Maximum length (seconds) of the training utterance, will filter out longer utterances
max_len=20

# Download directory for pre-trained models
download_dir=download/

# We suppose you have two TSV files: "data/raw/custom_train.tsv" and 
# "data/raw/custom_dev.tsv", where "custom" is your dataset name, 
# "train"/"dev" are used for training and validation respectively.

# Each line of the TSV files should be in one of the following formats:
# (1) `{uniq_id}\t{text}\t{wav_path}` if the text corresponds to the full wav,
# (2) `{uniq_id}\t{text}\t{wav_path}\t{start_time}\t{end_time}` if text corresponds
#     to part of the wav. The start_time and end_time specify the start and end
#     times of the text within the wav, which should be in seconds.
# > Note: {uniq_id} must be unique for each line.
# for subset in train dev;do
#       file_path=data/raw/custom_${subset}.tsv
#       [ -f "$file_path" ] || { echo "Error: expect $file_path !" >&2; exit 1; }
# done
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
      echo "Stage 9: install k2"
      pip install k2==1.24.4.dev20250208+cuda12.1.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html
      # https://github.com/k2-fsa/k2/blob/master/k2/python/k2/__init__.py#L13 delete the cuda version check
      RUN sed -i '/if (/,/^    )/d' /usr/local/lib/python3.12/dist-packages/k2/__init__.py
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Download pre-trained model, tokens file, and model config"
      # Uncomment this line to use HF mirror
      # export HF_ENDPOINT=https://hf-mirror.com
      hf_repo=k2-fsa/ZipVoice
      mkdir -p ${download_dir}
      for file in model.pt tokens.txt model.json; do
            huggingface-cli download \
                  --local-dir ${download_dir} \
                  ${hf_repo} \
                  zipvoice/${file}
      done
fi

### Training ZipVoice (5 - 6)

if [ $stage -le 41 ] && [ $stop_stage -ge 41 ]; then
    echo "Stage 4: Run inference"
    model_dir=exp/zipvoice_finetune
    model_dir=exp/zipvoice_finetune_ln_sigma
    python3 -m zipvoice.bin.infer_zipvoice_trt \
        --model-dir $model_dir \
        --checkpoint-name epoch-1.pt \
        --huggingface-dataset-name yuekai/seed_tts_cosy2 \
        --huggingface-dataset-split wenetspeech4tts \
        --batch-size 8 \
        --num-step 16 \
        --res-dir results_${model_dir}
fi

if [ $stage -le 42 ] && [ $stop_stage -ge 42 ]; then
  echo "stage 11: Test the model"
  datasets=(wenetspeech4tts zero_shot_zh test_zh)
  datasets=(zero_shot_zh)
  datasets=(wenetspeech4tts)
  model_dir=exp/zipvoice_finetune_ln_sigma
  for dataset in ${datasets[@]}; do

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

  guidance_scale=1.0
  model_name=zipvoice
  num_steps=16
  rollout_n=8
  enable_ln_sigma_sampling=True
  output_dir=results/${model_dir}_${dataset}_rollout_${rollout_n}
  python3 test_pipeline.py \
    --model-dir $model_dir \
    --checkpoint-name epoch-3.pt \
    --rollout-n ${rollout_n} \
    --enable-ln-sigma-sampling  \
    --huggingface-dataset-split ${dataset} \
    --batch-size 8 \
    --num-step 16 \
    --model-name ${model_name} \
    --results-dir results_${model_dir}
  done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Fine-tune the ZipVoice model"

      [ -z "$max_len" ] && { echo "Error: max_len is not set!" >&2; exit 1; }

      python3 -m zipvoice.bin.train_zipvoice \
            --world-size 4 \
            --use-fp16 1 \
            --finetune 1 \
            --base-lr 0.0001 \
            --num-iters 10000 \
            --save-every-n 1000 \
            --max-duration 500 \
            --max-len ${max_len} \
            --model-config ${download_dir}/zipvoice/model.json \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --token-file ${download_dir}/zipvoice/tokens.txt \
            --dataset aishell3 \
            --on-the-fly-feats True \
            --exp-dir exp/zipvoice_finetune

fi

if [ ${stage} -le 50 ] && [ ${stop_stage} -ge 50 ]; then
      echo "Stage 50: Fine-tune the ZipVoice model with ln_sigma head"

      [ -z "$max_len" ] && { echo "Error: max_len is not set!" >&2; exit 1; }

      python3 -m zipvoice.bin.train_zipvoice \
            --world-size 4 \
            --use-fp16 1 \
            --finetune 1 \
            --base-lr 0.0001 \
            --num-iters 10000 \
            --save-every-n 1000 \
            --max-duration 500 \
            --max-len ${max_len} \
            --model-config ${download_dir}/zipvoice/model.json \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --token-file ${download_dir}/zipvoice/tokens.txt \
            --dataset aishell3 \
            --on-the-fly-feats True \
            --enable-ln-sigma-head True \
            --only-train-ln-sigma-head True \
            --exp-dir exp/zipvoice_finetune_ln_sigma

fi

if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ]; then
      echo "Stage 51: Fine-tune the ZipVoice model with ln_sigma head"

      [ -z "$max_len" ] && { echo "Error: max_len is not set!" >&2; exit 1; }

      python3 -m zipvoice.bin.train_zipvoice \
            --world-size 4 \
            --use-fp16 1 \
            --finetune 1 \
            --base-lr 0.0001 \
            --num-iters 10000 \
            --save-every-n 1000 \
            --max-duration 500 \
            --max-len ${max_len} \
            --model-config ${download_dir}/zipvoice/model.json \
            --checkpoint exp/zipvoice_finetune_ln_sigma/epoch-1.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --token-file ${download_dir}/zipvoice/tokens.txt \
            --dataset aishell3 \
            --on-the-fly-feats True \
            --enable-ln-sigma-head True \
            --exp-dir exp/zipvoice_finetune_ln_sigma

fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Average the checkpoints for ZipVoice"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 10000 \
            --avg 2 \
            --model-name zipvoice \
            --exp-dir exp/zipvoice_finetune
      # The generated model is exp/zipvoice_finetune/iter-10000-avg-2.pt
fi

### Inference with PyTorch models (7)

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "Stage 7: Inference of the ZipVoice model"

      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice \
            --model-dir exp/zipvoice_finetune/ \
            --checkpoint-name iter-10000-avg-2.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --test-list test.tsv \
            --res-dir results/test_finetune\
            --num-step 16
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
      echo "Stage 8: RL Fine-tune the ZipVoice model using aishell 3 data"

      # [ -z "$max_len" ] && { echo "Error: max_len is not set!" >&2; exit 1; }
      noise_level=0.8
      num_steps=8
      exp_name=zipvoice_grpo_${noise_level}_${num_steps}_only_first_step
      python3 -m zipvoice.bin.train_zipvoice_grpo \
      --world-size 1 \
      --num-steps ${num_steps} \
      --train-batch-size 8 \
      --eval-batch-size 32 \
      --num-audio-per-prompt 8 \
      --num-batches-per-epoch 2 \
      --noise-level ${noise_level} \
      --global-std 1 \
      --learning-rate 1e-5 \
      --save-freq 100 \
      --eval-freq 10 \
      --huggingface-dataset-split wenetspeech4tts \
      --use-fp16 1 \
      --run-name ${exp_name} \
      --exp-dir exp/${exp_name} \
      --pretrained-model zipvoice_distill \
      --dataset-path aishell-3-cosy.jsonl

fi



if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  echo "stage 10: start token2wav asr server for reward function"

#   git clone https://github.com/yuekaizhang/PytritonSenseVoice.git /workspace/PytritonSenseVoice
#   cd /workspace/PytritonSenseVoice
#   pip install -e .
  # pip install jiwer WeTextProcessing wandb zhon sherpa-onnx
  n_gpus=1
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 reward_server.py --number-of-devices $n_gpus

fi 

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
  echo "stage 11: Test the model"
  datasets=(wenetspeech4tts zero_shot_zh test_zh)
  datasets=(zero_shot_zh)
  datasets=(wenetspeech4tts)
  # datasets=(test_zh)
  for dataset in ${datasets[@]}; do

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  noise_levels=(0.4 0.5 0.6 0.7)
  noise_levels=(0.8)
  guidance_scale=1.0
  model_name=zipvoice
  num_steps=16
  rollout_n=8
  for noise_level in ${noise_levels[@]}; do
  output_dir=results/${model_name}_only_first_step_${dataset}_noise_${noise_level}_step_${num_steps}_rollout_${rollout_n}_guidance_${guidance_scale}_fixed_intial_noise
  python3 test_pipeline.py \
    --model-name ${model_name} \
    --world-size 8 \
    --num-step ${num_steps} \
    --results-dir $output_dir \
    --batch-size 4 \
    --guidance-scale ${guidance_scale} \
    --rollout-n ${rollout_n} \
    --noise-level ${noise_level} \
    --huggingface-dataset-split ${dataset}
  bash scripts/compute_wer.sh $output_dir ${dataset}
  done
  done
fi