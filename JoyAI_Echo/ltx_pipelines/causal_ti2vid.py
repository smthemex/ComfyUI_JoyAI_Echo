"""Autoregressive text/image-to-video rollout for Echo-WM Flash."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ..ltx_core.model.video_vae import decode_video as vae_decode_video
from ..ltx_core.model.video_vae.video_vae import decode_video_to_fhwc
from ..ltx_core.model.video_vae.tiling import TilingConfig
from ..ltx_core.quantization import QuantizationPolicy
from ..ltx_core.tools import AudioLatentTools
from ..ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ..ltx_causal import (
    DEFAULT_CAUSAL_TIMESTEPS,
    CausalCacheConfig,
    CausalModelWrapper,
    causal_audio_frames,
    causal_rollout,
    causal_video_blocks,
)
from .utils import ModelLedger, assert_resolution, cleanup_memory, combined_image_conditionings, encode_prompts, get_device
from .utils.args import ImageConditioningInput
from .utils.helpers import create_noised_state, noise_video_state
from .utils.types import PipelineComponents
from ..utils import streaming_single_model,streaming_prefetch_model,_full_gpu_ctx,streaming_single_fast
from ..ltx_core.model.transformer.model import BlockGPUManager
device = get_device()


class CausalTI2VidPipeline:
    """Inference-only 4-step autoregressive I2V pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
        action_config=None,
        cache_config: CausalCacheConfig = CausalCacheConfig(),
    ) -> None:
        self.dtype = torch.bfloat16
        self.device = device
        self.action_config = action_config
        cache_config.validate()
        self.cache_config = cache_config
        self.streaming_mode="fast"
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,
            quantization=quantization,
            load_model="origin",
            action_config=self.action_config,
            
        )
        self.pipeline_components = PipelineComponents(dtype=self.dtype, device=device)

    def _model_ctx(self,model,prefetch_count: int | None,) :
        if prefetch_count is not None :
            layers_attr="model.transformer_blocks"
            if self.streaming_mode=="fast":
                return streaming_single_fast(
                    model,
                    layers_attr=layers_attr,
                    target_device=torch.device("cuda"),
                )
            elif self.streaming_mode=="slow":
                    return streaming_single_model(
                        model,
                        layers_attr=layers_attr,
                        target_device=torch.device("cuda"),
                    )
            elif self.streaming_mode=="auto":
                return streaming_prefetch_model(
                    model,
                    layers_attr=layers_attr,
                    target_device=torch.device("cuda"),
                    prefetch_count=prefetch_count,
                )
            else:
                gpu_manager=BlockGPUManager(block_group_size=prefetch_count)
                gpu_manager.setup_for_inference(model.model.velocity_model)
                model.gpu_manager=gpu_manager
                return _full_gpu_ctx(model)
        
        return _full_gpu_ctx(model)

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        action_cond: dict[str, torch.Tensor],
        timesteps: tuple[int, ...] | list[int] = DEFAULT_CAUSAL_TIMESTEPS,
        video_tiling_config: TilingConfig | None = None,
        te_cond=None,
        prefetch_count: int | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=False)
        latent_frames = (num_frames - 1) // 8 + 1
        if num_frames != (latent_frames - 1) * 8 + 1:
            raise ValueError("causal --num-frames must be 1 + 8*n output frames")
        causal_video_blocks(latent_frames, self.cache_config.video_chunk_size)
        # The causal student is trained on one positive conditioning branch.
        if te_cond is  None:
            encoded_prompt, = encode_prompts([prompt], self.model_ledger)
            if encoded_prompt.audio_encoding is None:
                raise ValueError("the causal AV checkpoint must provide audio text embeddings")
            video_context=encoded_prompt.video_encoding
            audio_context=encoded_prompt.audio_encoding
            context_mask=encoded_prompt.attention_mask
        else:
            video_context=te_cond["video_context"]
            audio_context=te_cond["audio_context"]
            context_mask=te_cond["attention_mask"]

        output_shape = VideoPixelShape(1, num_frames, height, width, frame_rate)
        video_encoder = self.model_ledger.video_encoder()
        conditionings = combined_image_conditionings(
            images, height, width, video_encoder, self.dtype, self.device
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        del video_encoder
        print("video encoder is done")
        cleanup_memory()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator)
        video_state, video_tools = noise_video_state(
            output_shape, noiser, conditionings, self.pipeline_components,
            self.dtype, self.device,
        )
        audio_frames = causal_audio_frames(latent_frames, self.cache_config.video_chunk_size)
        audio_shape = AudioLatentShape(batch=1, channels=8, frames=audio_frames, mel_bins=16)
        audio_tools = AudioLatentTools(self.pipeline_components.audio_patchifier, audio_shape)
        audio_state = create_noised_state(
            audio_tools, [], noiser, self.dtype, self.device
        )
        print("audio encoder is done,start transformer ")
        x0_model = self.model_ledger.transformer(action_config=self.action_config)
        wrapper = CausalModelWrapper(
            x0_model.velocity_model,
            patches_per_frame=(height // 32) * (width // 32),
            cache=self.cache_config,
        )
        with self._model_ctx(wrapper,prefetch_count) as wrapper:
            generated_video, generated_audio = causal_rollout(
                wrapper=wrapper,
                clean_video=video_state.clean_latent,
                clean_audio=audio_state.clean_latent,
                video_positions=video_state.positions,
                audio_positions=audio_state.positions,
                video_context=video_context,
                audio_context=audio_context,
                context_mask=context_mask,
                action_cond=action_cond,
                seed=seed,
                timesteps=timesteps,
            )
        del wrapper, x0_model
        cleanup_memory()

        video_state = video_tools.unpatchify(video_tools.clear_conditioning(
            video_state.__class__(
                latent=generated_video,
                denoise_mask=video_state.denoise_mask,
                positions=video_state.positions,
                clean_latent=video_state.clean_latent,
                attention_mask=None,
            )
        ))
        audio_state = audio_tools.unpatchify(audio_tools.clear_conditioning(
            audio_state.__class__(
                latent=generated_audio,
                denoise_mask=audio_state.denoise_mask,
                positions=audio_state.positions,
                clean_latent=audio_state.clean_latent,
                attention_mask=None,
            )
        ))
        decoded_video = decode_video_to_fhwc(
            video_state.latent,
            self.model_ledger.video_decoder(),
            tiling_config=video_tiling_config,
            generator=generator,
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent,
            self.model_ledger.audio_decoder(),
            self.model_ledger.vocoder(),
        )
        return decoded_video, decoded_audio
