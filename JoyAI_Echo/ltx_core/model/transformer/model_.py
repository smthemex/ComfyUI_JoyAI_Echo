from enum import Enum

import torch

from ...guidance.perturbations import BatchedPerturbationConfig
from .adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from .attention_ import AttentionCallable, AttentionFunction
from .modality import Modality
from .rope_ import LTXRopeType
from .transformer_ import BasicAVTransformerBlock, TransformerConfig
from .transformer_args_ import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ...utils import to_denoised


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
            apply_gated_attention=apply_gated_attention,
        )

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: AttentionFunction | AttentionCallable,
        apply_gated_attention: bool,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    num_layers=num_layers,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )

    def enable_action_conditioning(self, action_config) -> None:
        """Attach the optional pure-UCPE branch after base weights are loaded."""
        from .transformer_ import ActionBlockConfig

        if not isinstance(action_config, ActionBlockConfig):
            raise TypeError(f"action_config must be ActionBlockConfig, got {type(action_config)}")
        if not self.model_type.is_video_enabled():
            raise RuntimeError("UCPE conditioning requires a video-enabled model")
        video_config = TransformerConfig(
            dim=self.inner_dim,
            heads=self.num_attention_heads,
            d_head=self.inner_dim // self.num_attention_heads,
            context_dim=0,
        )
        for block in self.transformer_blocks:
            block._init_action_params(video_config, action_config)

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
        action_cond: dict | None = None,
        kv_caches: list[dict] | None = None,
        current_video_token_start: int = 0,
        current_audio_token_start: int = 0,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks for LTXAV."""

        ucpe_viewmats = action_cond.get("ucpe_viewmats") if action_cond else None
        ucpe_Ks = action_cond.get("ucpe_Ks") if action_cond else None
        for layer_index, block in enumerate(self.transformer_blocks):
            layer_cache = kv_caches[layer_index] if kv_caches is not None else None
            if self._enable_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training.
                # With use_reentrant=False, we can pass dataclasses directly -
                # PyTorch will track all tensor leaves in the computation graph.
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    perturbations,
                    ucpe_viewmats,
                    ucpe_Ks,
                    layer_cache,
                    current_video_token_start,
                    current_audio_token_start,
                    use_reentrant=False,
                )
            else:
                video, audio = block(
                    video=video,
                    audio=audio,
                    perturbations=perturbations,
                    ucpe_viewmats=ucpe_viewmats,
                    ucpe_Ks=ucpe_Ks,
                    kv_cache=layer_cache,
                    current_video_token_start=current_video_token_start,
                    current_audio_token_start=current_audio_token_start,
                )

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig,
        action_cond: dict | None = None,
        kv_caches: list[dict] | None = None,
        current_video_token_start: int = 0,
        current_audio_token_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
            action_cond=action_cond,
            kv_caches=kv_caches,
            current_video_token_start=current_video_token_start,
            current_audio_token_start=current_audio_token_start,
        )

        # Process output
        vx = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax

    def init_av_kv_caches(  # noqa: PLR0913
        self,
        batch_size: int,
        max_video_tokens: int,
        max_audio_tokens: int,
        text_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        video_local_attn_tokens: int = -1,
        video_sink_tokens: int = 0,
        video_ucpe_local_attn_tokens: int | None = None,
        video_ucpe_sink_tokens: int | None = None,
        audio_local_attn_tokens: int = -1,
        audio_sink_tokens: int = 0,
    ) -> list[dict]:
        """Allocate inference-only per-layer AV caches."""

        def allocate(max_tokens: int, dim: int, local: int, sink: int) -> dict:
            capacity = max_tokens if local < 0 else min(max_tokens, local)
            if capacity <= 0:
                raise ValueError("KV cache capacity must be positive")
            return {
                "k": torch.zeros(batch_size, capacity, dim, device=device, dtype=dtype),
                "v": torch.zeros(batch_size, capacity, dim, device=device, dtype=dtype),
                "positions": torch.full((capacity,), -1, device=device, dtype=torch.long),
                "length": 0,
                "local_attn_size": local,
                "sink_tokens": sink,
            }

        def allocate_text(dim: int) -> dict:
            return {
                "k": torch.zeros(batch_size, text_seq_len, dim, device=device, dtype=dtype),
                "v": torch.zeros(batch_size, text_seq_len, dim, device=device, dtype=dtype),
                "length": 0,
                "is_init": False,
            }

        ucpe_local = video_local_attn_tokens if video_ucpe_local_attn_tokens is None else video_ucpe_local_attn_tokens
        ucpe_sink = video_sink_tokens if video_ucpe_sink_tokens is None else video_ucpe_sink_tokens
        caches: list[dict] = []
        for block in self.transformer_blocks:
            layer: dict = {}
            if self.model_type.is_video_enabled():
                layer["video_self"] = allocate(max_video_tokens, self.inner_dim, video_local_attn_tokens, video_sink_tokens)
                layer["video_text"] = allocate_text(self.inner_dim)
                if block.action_ucpe_enabled:
                    layer["video_ucpe"] = allocate(
                        max_video_tokens,
                        block.ucpe_num_heads * block.ucpe_head_dim,
                        ucpe_local,
                        ucpe_sink,
                    )
            if self.model_type.is_audio_enabled() and max_audio_tokens > 0:
                layer["audio_self"] = allocate(max_audio_tokens, self.audio_inner_dim, audio_local_attn_tokens, audio_sink_tokens)
                layer["audio_text"] = allocate_text(self.audio_inner_dim)
            if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled() and max_audio_tokens > 0:
                layer["a2v"] = allocate(max_audio_tokens, self.audio_inner_dim, audio_local_attn_tokens, audio_sink_tokens)
                layer["v2a"] = allocate(max_video_tokens, self.audio_inner_dim, video_local_attn_tokens, video_sink_tokens)
            caches.append(layer)
        return caches


class LegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
        action_cond: dict | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations, action_cond=action_cond)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        action_cond: dict | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations, action_cond=action_cond)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
