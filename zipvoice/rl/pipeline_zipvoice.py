import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from vocos import Vocos

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import load_prompt_wav, rms_norm


HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}


class ZipVoicePipeline(object):
    def __init__(
        self,
        model_name: str = None,
        model_dir: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
        vocoder_path: Optional[str] = None,
        tokenizer_type: str = "emilia",
        lang: str = "en-us",
        device: torch.device = torch.device("cuda"),
    ):
        params = AttributeDict()

        if model_dir is not None:
            params.model_dir = Path(model_dir)
            if not params.model_dir.is_dir():
                raise FileNotFoundError(f"{params.model_dir} does not exist")
            if checkpoint_name is None:
                checkpoint_name = "model.pt"
            for filename in [checkpoint_name, "model.json", "tokens.txt"]:
                if not (params.model_dir / filename).is_file():
                    raise FileNotFoundError(f"{params.model_dir / filename} does not exist")
            model_ckpt = params.model_dir / checkpoint_name
            model_config_file = params.model_dir / "model.json"
            token_file = params.model_dir / "tokens.txt"
            logging.info(
                f"Using {model_name} in local model dir {params.model_dir}, "
                f"checkpoint {checkpoint_name}"
            )
        else:
            logging.info(f"Using pretrained {model_name} model from Huggingface")
            model_ckpt = hf_hub_download(
                HUGGINGFACE_REPO, filename=f"{MODEL_DIR[model_name]}/model.pt"
            )
            model_config_file = hf_hub_download(
                HUGGINGFACE_REPO, filename=f"{MODEL_DIR[model_name]}/model.json"
            )
            token_file = hf_hub_download(
                HUGGINGFACE_REPO, filename=f"{MODEL_DIR[model_name]}/tokens.txt"
            )

        if tokenizer_type == "emilia":
            self.tokenizer = EmiliaTokenizer(token_file=token_file)
        elif tokenizer_type == "libritts":
            self.tokenizer = LibriTTSTokenizer(token_file=token_file)
        elif tokenizer_type == "espeak":
            self.tokenizer = EspeakTokenizer(token_file=token_file, lang=lang)
        else:
            assert tokenizer_type == "simple"
            self.tokenizer = SimpleTokenizer(token_file=token_file)

        tokenizer_config = {"vocab_size": self.tokenizer.vocab_size, "pad_id": self.tokenizer.pad_id}

        with open(model_config_file, "r") as f:
            model_config = json.load(f)

        if model_name == "zipvoice":
            self.model = ZipVoice(
                **model_config["model"],
                **tokenizer_config,
            )
        else:
            assert model_name == "zipvoice_distill"
            self.model = ZipVoiceDistill(
                **model_config["model"],
                **tokenizer_config,
            )

        load_checkpoint(filename=model_ckpt, model=self.model, strict=True)

        # if "cuda" in device and torch.cuda.is_available():
        #     self.device = torch.device(device)
        # elif "mps" in device and torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        # else:
        #     self.device = torch.device("cpu")
        self.device = device
        logging.info(f"Device: {self.device}")

        self.model = self.model.to(self.device)
        self.model.eval()

        if vocoder_path:
            self.vocoder = Vocos.from_hparams(f"{vocoder_path}/config.yaml")
            state_dict = torch.load(
                f"{vocoder_path}/pytorch_model.bin",
                weights_only=True,
                map_location="cpu",
            )
            self.vocoder.load_state_dict(state_dict)
        else:
            self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")

        self.vocoder = self.vocoder.to(self.device)
        self.vocoder.eval()

        if model_config["feature"]["type"] == "vocos":
            self.feature_extractor = VocosFbank()
        else:
            raise NotImplementedError(
                f"Unsupported feature type: {model_config['feature']['type']}"
            )
        self.sampling_rate = model_config["feature"]["sampling_rate"]

        self.model_name = model_name
        model_defaults = {
            "zipvoice": {
                "num_step": 16,
                "guidance_scale": 1.0,
            },
            "zipvoice_distill": {
                "num_step": 8,
                "guidance_scale": 3.0,
            },
        }
        self.defaults = model_defaults.get(self.model_name, {})

    # @torch.inference_mode()
    def __call__(
        self,
        prompt_text: Union[str, List[str]],
        prompt_wav: Union[str, List[str], torch.Tensor, List[torch.Tensor]],
        text: Union[str, List[str]],
        num_step: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        speed: float = 1.0,
        t_shift: float = 0.5,
        target_rms: float = 0.1,
        feat_scale: float = 0.1,
        enable_sde: bool = True,
        sde_noise_level: float = 0.2,
        enable_ln_sigma_sampling: bool = False,
    ) -> List[torch.Tensor]:
        if num_step is None:
            num_step = self.defaults.get("num_step", 16)
        if guidance_scale is None:
            guidance_scale = self.defaults.get("guidance_scale", 1.0)

        prepared_inputs = self.prepare_latents(
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            text=text,
            target_rms=target_rms,
            feat_scale=feat_scale,
        )
        
        # NOTE: The original model.sample takes latents with name `noise`.
        # Here we follow diffusers convention and call it latents.
        # Inside ZipVoice.sample, it's x0, not used as noise for xt calculation.
        # But for ZipVoiceDistill, it's used as noise.
        # So we just pass it as `noise`.
        model_output = self.model.sample(
            tokens=prepared_inputs["tokens"],
            prompt_tokens=prepared_inputs["prompt_tokens"],
            prompt_features=prepared_inputs["prompt_features"],
            prompt_features_lens=prepared_inputs["prompt_features_lens"],
            speed=speed,
            t_shift=t_shift,
            duration="predict",
            num_step=num_step,
            guidance_scale=guidance_scale,
            enable_sde=enable_sde,
            sde_noise_level=sde_noise_level,
            enable_ln_sigma_sampling=enable_ln_sigma_sampling,
        )

        # if enable_sde:
        (
            pred_features,
            pred_features_lens,
            _,
            _,
            log_probs,
            latents,
            timesteps,
        ) = model_output
        # else:
        #     (
        #         pred_features,
        #         pred_features_lens,
        #         _,
        #         _,
        #     ) = model_output


        pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)
        
        batch_wavs = []
        prompt_rms_list = prepared_inputs["prompt_rms_list"]
        for i in range(pred_features.size(0)):
            wav = (
                self.vocoder.decode(pred_features[i][None, :, : pred_features_lens[i]])
                .squeeze(1)
                .clamp(-1, 1)
            )
            if prompt_rms_list[i] < target_rms:
                wav = wav * prompt_rms_list[i] / target_rms
            batch_wavs.append(wav.cpu())
        return batch_wavs, latents, log_probs, timesteps
        # if enable_sde:
        #     return batch_wavs, latents, log_probs, timesteps
        # else: # sde disabled
        #     return batch_wavs

    # @torch.inference_mode()
    def prepare_latents(
        self,
        prompt_text: Union[str, List[str]],
        prompt_wav: Union[str, List[str], torch.Tensor, List[torch.Tensor]],
        text: Union[str, List[str]],
        target_rms: float = 0.1,
        feat_scale: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(prompt_text, str):
            prompt_text = [prompt_text]
        if isinstance(text, str):
            text = [text]
        
        if isinstance(prompt_wav, list) and isinstance(prompt_wav[0], str): # list of paths
            prompt_wavs_list = [load_prompt_wav(p, sampling_rate=self.sampling_rate) for p in prompt_wav]
        elif isinstance(prompt_wav, str): # single path
            prompt_wavs_list = [load_prompt_wav(prompt_wav, sampling_rate=self.sampling_rate)]
        elif isinstance(prompt_wav, torch.Tensor): # single tensor
            prompt_wavs_list = [prompt_wav]
        elif isinstance(prompt_wav, list) and isinstance(prompt_wav[0], torch.Tensor):
            prompt_wavs_list = prompt_wav
        else:
            raise ValueError("Unsupported type for prompt_wav")

        prompt_features_list = []
        prompt_rms_list = []
        for p_wav in prompt_wavs_list:
            p_wav, prompt_rms = rms_norm(p_wav, target_rms)
            prompt_rms_list.append(prompt_rms)
            prompt_features = self.feature_extractor.extract(
                p_wav, sampling_rate=self.sampling_rate
            ).to(self.device)
            prompt_features_list.append(prompt_features)
        
        prompt_features_lens = torch.tensor(
            [pf.size(0) for pf in prompt_features_list], device=self.device
        )
        prompt_features = torch.nn.utils.rnn.pad_sequence(
            prompt_features_list, batch_first=True, padding_value=0.0
        )
        prompt_features = prompt_features * feat_scale

        tokens = self.tokenizer.texts_to_token_ids(text)
        prompt_tokens = self.tokenizer.texts_to_token_ids(prompt_text)

        return {
            "tokens": tokens,
            "prompt_tokens": prompt_tokens,
            "prompt_features": prompt_features,
            "prompt_features_lens": prompt_features_lens,
            "prompt_rms_list": prompt_rms_list,
        }