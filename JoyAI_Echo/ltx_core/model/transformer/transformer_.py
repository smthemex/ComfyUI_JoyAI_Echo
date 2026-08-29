from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F

from ...guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from .adaln import adaln_embedding_coefficient
from .attention_ import Attention, AttentionCallable, AttentionFunction, update_kv_cache
from .feed_forward import FeedForward
from .rope_ import LTXRopeType
from .transformer_args_ import TransformerArgs
from .ucpe_prope import _prepare_apply_fns
from ...utils import rms_norm


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


@dataclass
class ActionBlockConfig:
    """Configuration for the optional pure-UCPE camera branch."""

    enabled: bool = False
    block_indices: list[int] = field(default_factory=list)
    ucpe: bool = True
    ucpe_attn_dim: int | None = None
    ucpe_num_heads: int | None = None
    ucpe_patches_x: int = 40
    ucpe_patches_y: int = 22
    ucpe_image_width: int = 1280
    ucpe_image_height: int = 704
    ucpe_freq_base: float = 100.0
    ucpe_freq_scale: float = 1.0

    def owns(self, block_idx: int) -> bool:
        return self.enabled and self.ucpe and block_idx in self.block_indices


def active_sink_fifo_indices(
    current_end: int, local_size: int, sink_size: int, device: torch.device
) -> tuple[torch.Tensor, int]:
    """Indices represented by a bounded ``sink + recent FIFO`` cache."""
    if local_size <= 0 or sink_size < 0 or sink_size >= local_size:
        raise ValueError(f"invalid sink/FIFO layout: local={local_size}, sink={sink_size}")
    if current_end <= local_size:
        return torch.arange(current_end, device=device), 0
    recent_start = max(sink_size, current_end - (local_size - sink_size))
    return torch.cat((torch.arange(sink_size, device=device), torch.arange(recent_start, current_end, device=device))), recent_start


def rebase_viewmat_translation(viewmats: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Apply one common right-side translation, preserving relative cameras."""
    with torch.autocast(device_type=viewmats.device.type, enabled=False):
        matrices = viewmats.float()
        anchor = anchor.float()
        shift = -(anchor[..., :3, :3].transpose(-1, -2) @ anchor[..., :3, 3:4])
        result = matrices.clone()
        result[..., :3, 3:4] += result[..., :3, :3] @ shift
    return result


def _ucpe_transform(apply_fn, value: torch.Tensor) -> torch.Tensor:
    dtype = value.dtype
    with torch.autocast(device_type=value.device.type, enabled=False):
        return apply_fn(value.float()).to(dtype)


def _ucpe_cache_attend(cache: dict, start: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, seq, dim = k.shape
    flat_k = k.transpose(1, 2).reshape(batch, seq, heads * dim)
    flat_v = v.transpose(1, 2).reshape(batch, seq, heads * dim)
    flat_k, flat_v = update_kv_cache(cache, start, flat_k, flat_v)
    active = flat_k.shape[1]
    return (
        flat_k.view(batch, active, heads, dim).transpose(1, 2),
        flat_v.view(batch, active, heads, dim).transpose(1, 2),
    )


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        num_layers: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
    ):
        super().__init__()

        self.idx = idx
        self.num_layers = num_layers
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            video_sst_size = adaln_embedding_coefficient(video.cross_attention_adaln)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(video_sst_size, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            audio_sst_size = adaln_embedding_coefficient(audio.cross_attention_adaln)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(audio_sst_size, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )

        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps
        self.action_owns = False
        self.action_ucpe_enabled = False

    def _init_action_params(self, video: TransformerConfig, action_config: ActionBlockConfig) -> None:
        """Attach the zero-initialized pure-UCPE branch to this block."""
        if self.action_owns:
            raise RuntimeError(f"Action params already initialized for block idx={self.idx}")
        if not action_config.owns(self.idx):
            return
        from .ucpe_prope import PropeDotProductAttention

        self.action_owns = True
        self.action_ucpe_enabled = True
        vdim = video.dim
        attn_dim = action_config.ucpe_attn_dim or vdim
        num_heads = action_config.ucpe_num_heads or video.heads
        if attn_dim % num_heads != 0 or (attn_dim // num_heads) % 4 != 0:
            raise ValueError("UCPE attention dimension must be divisible by heads and by 4")
        self.ucpe_num_heads = num_heads
        self.ucpe_head_dim = attn_dim // num_heads
        self.ucpe_q_proj = torch.nn.Linear(vdim, attn_dim, bias=False)
        self.ucpe_k_proj = torch.nn.Linear(vdim, attn_dim, bias=False)
        self.ucpe_v_proj = torch.nn.Linear(vdim, attn_dim, bias=False)
        self.ucpe_out_proj = torch.nn.Linear(attn_dim, vdim, bias=True)
        torch.nn.init.xavier_uniform_(self.ucpe_q_proj.weight)
        torch.nn.init.xavier_uniform_(self.ucpe_k_proj.weight)
        torch.nn.init.xavier_uniform_(self.ucpe_v_proj.weight)
        torch.nn.init.zeros_(self.ucpe_out_proj.weight)
        torch.nn.init.zeros_(self.ucpe_out_proj.bias)
        self.ucpe_prope = PropeDotProductAttention(
            head_dim=self.ucpe_head_dim,
            patches_x=action_config.ucpe_patches_x,
            patches_y=action_config.ucpe_patches_y,
            image_width=action_config.ucpe_image_width,
            image_height=action_config.ucpe_image_height,
            freq_base=action_config.ucpe_freq_base,
            freq_scale=action_config.ucpe_freq_scale,
        )

    def _apply_ucpe_attention(
        self,
        norm_vx: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        kv_cache: dict | None = None,
        kv_cache_start: int = 0,
    ) -> torch.Tensor:
        batch, seq_len, _ = norm_vx.shape
        heads, head_dim = self.ucpe_num_heads, self.ucpe_head_dim
        q = self.ucpe_q_proj(norm_vx).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        k = self.ucpe_k_proj(norm_vx).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        v = self.ucpe_v_proj(norm_vx).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        if kv_cache is not None and kv_cache.get("bounded_anchor_translation", False):
            ppf = int(kv_cache["patches_per_frame"])
            k, v = _ucpe_cache_attend(kv_cache, kv_cache_start, k, v)
            current_start = kv_cache_start // ppf
            current_end = current_start + seq_len // ppf
            indices, anchor_index = active_sink_fifo_indices(
                current_end,
                int(kv_cache["local_attn_size"]) // ppf,
                int(kv_cache["sink_tokens"]) // ppf,
                kv_cache["full_ucpe_viewmats"].device,
            )
            all_viewmats = kv_cache["full_ucpe_viewmats"]
            all_Ks = kv_cache["full_ucpe_Ks"]
            anchor = all_viewmats[:, anchor_index : anchor_index + 1]
            q_viewmats = rebase_viewmat_translation(all_viewmats[:, current_start:current_end], anchor)
            k_viewmats = rebase_viewmat_translation(all_viewmats.index_select(1, indices), anchor)
            kwargs = dict(
                head_dim=self.ucpe_prope.head_dim,
                patches_x=self.ucpe_prope.patches_x,
                patches_y=self.ucpe_prope.patches_y,
                image_width=self.ucpe_prope.image_width,
                image_height=self.ucpe_prope.image_height,
                coeffs_x=None if self.ucpe_prope.coeffs_x_0 is None else (self.ucpe_prope.coeffs_x_0, self.ucpe_prope.coeffs_x_1),
                coeffs_y=None if self.ucpe_prope.coeffs_y_0 is None else (self.ucpe_prope.coeffs_y_0, self.ucpe_prope.coeffs_y_1),
            )
            apply_q, _, apply_out = _prepare_apply_fns(viewmats=q_viewmats, Ks=all_Ks[:, current_start:current_end].float(), **kwargs)
            _, apply_kv, _ = _prepare_apply_fns(viewmats=k_viewmats, Ks=all_Ks.index_select(1, indices).float(), **kwargs)
            q = _ucpe_transform(apply_q, q)
            k = _ucpe_transform(apply_kv, k)
            v = _ucpe_transform(apply_kv, v)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            out = _ucpe_transform(apply_out, out)
        else:
            # Preserve the original base-inference dtype/autocast behavior.
            self.ucpe_prope._precompute_and_cache_apply_fns(viewmats=viewmats, Ks=Ks)
            q = self.ucpe_prope._apply_to_q(q)
            k = self.ucpe_prope._apply_to_kv(k)
            v = self.ucpe_prope._apply_to_kv(v)
            if kv_cache is not None:
                k, v = _ucpe_cache_attend(kv_cache, kv_cache_start, k, v)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            out = self.ucpe_prope._apply_to_o(out)
        out = out.to(dtype=self.ucpe_out_proj.weight.dtype)
        return self.ucpe_out_proj(out.transpose(1, 2).reshape(batch, seq_len, heads * head_dim))

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, scale_shift_indices
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale, shift = (t.squeeze(2) for t in scale_shift_ada_values)
        (gate,) = (t.squeeze(2) for t in gate_ada_values)

        return scale, shift, gate

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation."""
        if cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x.shape[0], timestep, slice(6, 9))
            return apply_cross_attention_adaln(
                x,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
                self.norm_eps,
                crossattn_cache,
            )
        return attn(
            rms_norm(x, eps=self.norm_eps), context=context, mask=context_mask,
            crossattn_cache=crossattn_cache,
        )

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
        ucpe_viewmats: torch.Tensor | None = None,
        ucpe_Ks: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        current_video_token_start: int = 0,
        current_audio_token_start: int = 0,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        batch_size = (video or audio).x.shape[0]

        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            del vshift_msa, vscale_msa

            all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)
            none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)
            v_mask = (
                perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                if not all_perturbed and not none_perturbed
                else None
            )
            attn_out = self.attn1(
                    norm_vx,
                    pe=video.positional_embeddings,
                    mask=video.self_attention_mask,
                    perturbation_mask=v_mask,
                    all_perturbed=all_perturbed,
                    kv_cache=kv_cache.get("video_self") if kv_cache else None,
                    kv_cache_start=current_video_token_start,
            )
            if self.action_ucpe_enabled and ucpe_viewmats is not None and ucpe_Ks is not None:
                attn_out = attn_out + self._apply_ucpe_attention(
                    norm_vx, ucpe_viewmats, ucpe_Ks,
                    kv_cache=kv_cache.get("video_ucpe") if kv_cache else None,
                    kv_cache_start=current_video_token_start,
                )
            vx = vx + attn_out * vgate_msa
            del vgate_msa, norm_vx, v_mask, attn_out
            vx = vx + self._apply_text_cross_attention(
                vx,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
                crossattn_cache=kv_cache.get("video_text") if kv_cache else None,
            )

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            del ashift_msa, ascale_msa
            all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx)
            none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx)
            a_mask = (
                perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                if not all_perturbed and not none_perturbed
                else None
            )
            audio_self_attention_mask = audio.self_attention_mask
            if self.idx >= int(self.num_layers * 0.7):
                audio_self_attention_mask = audio.late_self_attention_mask
            ax = (
                ax
                + self.audio_attn1(
                    norm_ax,
                    pe=audio.positional_embeddings,
                    mask=audio_self_attention_mask,
                    perturbation_mask=a_mask,
                    all_perturbed=all_perturbed,
                    kv_cache=kv_cache.get("audio_self") if kv_cache else None,
                    kv_cache_start=current_audio_token_start,
                )
                * agate_msa
            )
            del agate_msa, norm_ax, a_mask
            ax = ax + self._apply_text_cross_attention(
                ax,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
                crossattn_cache=kv_cache.get("audio_text") if kv_cache else None,
            )

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            if run_a2v and not perturbations.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx):
                scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                del scale_ca_video_a2v, shift_ca_video_a2v

                scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                del scale_ca_audio_a2v, shift_ca_audio_a2v
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                cross_attention_mask = video.cross_attention_mask
                cross_output_mask = video.cross_output_mask
                if self.idx >= int(self.num_layers * 0.7):
                    if video.late_cross_attention_mask is not None:
                        cross_attention_mask = video.late_cross_attention_mask
                    if video.late_cross_output_mask is not None:
                        cross_output_mask = video.late_cross_output_mask
                cross_output_mask = cross_output_mask if cross_output_mask is not None else 1.0
                vx = vx + (
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        mask=cross_attention_mask,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                        kv_cache=kv_cache.get("a2v") if kv_cache else None,
                        kv_cache_start=current_audio_token_start,
                    )
                    * gate_out_a2v
                    * a2v_mask
                    * cross_output_mask
                )
                del gate_out_a2v, a2v_mask, vx_scaled, ax_scaled, cross_output_mask

            if run_v2a and not perturbations.all_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx):
                scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                del scale_ca_audio_v2a, shift_ca_audio_v2a
                scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                del scale_ca_video_v2a, shift_ca_video_v2a
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                cross_attention_mask = audio.cross_attention_mask
                cross_output_mask = audio.cross_output_mask
                if self.idx >= int(self.num_layers * 0.7):
                    if audio.late_cross_attention_mask is not None:
                        cross_attention_mask = audio.late_cross_attention_mask
                    if audio.late_cross_output_mask is not None:
                        cross_output_mask = audio.late_cross_output_mask
                cross_output_mask = cross_output_mask if cross_output_mask is not None else 1.0
                v2a_update = (
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        mask=cross_attention_mask,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                        kv_cache=kv_cache.get("v2a") if kv_cache else None,
                        kv_cache_start=current_video_token_start,
                    )
                    * gate_out_v2a
                    * v2a_mask
                    * cross_output_mask
                )
                v2a_grad_scale = float(getattr(audio, "v2a_grad_scale", 1.0))
                if v2a_grad_scale != 1.0 and torch.is_grad_enabled():
                    v2a_update = v2a_update.detach() + v2a_grad_scale * (v2a_update - v2a_update.detach())
                ax = ax + v2a_update
                del gate_out_v2a, v2a_mask, ax_scaled, vx_scaled, cross_output_mask, v2a_update

            del vx_norm3, ax_norm3

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

            del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    crossattn_cache: dict | None = None,
) -> torch.Tensor:
    batch_size = x.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = rms_norm(x, eps=norm_eps) * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(
        attn_input,
        context=encoder_hidden_states,
        mask=context_mask,
        crossattn_cache=crossattn_cache,
    ) * q_gate
