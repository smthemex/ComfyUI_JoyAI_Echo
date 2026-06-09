"""Unified inference entrypoint: load models once, process all prompt files."""

from __future__ import annotations

import json
import soundfile as sf
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any
from tqdm import tqdm
# Ensure local packages are importable when running from repo root
# _REPO_ROOT = Path(__file__).resolve().parent
# for _subpath in ["ltx-core/src", "ltx-pipelines/src", "ltx-distillation/src"]:
#     _p = str(_REPO_ROOT / _subpath)
#     if _p not in sys.path:
#         sys.path.insert(0, _p)

import torch
#import torchaudio
import yaml

from .ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
# from .ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
# from .ltx_distillation.inference.memory_bidirectional_pipeline import BidirectionalMemoryAVInferencePipeline
from .ltx_distillation.inference.memory_multishot import (
    PairedAudioVideoMemoryBank,
    audio_waveform_stats,
    build_paired_audio_memory_kwargs,
    load_multishot_prompts,
    video_uint8_to_pil_frames,
)
from .ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
from .ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
from .ltx_distillation.models.vae_wrapper import create_vae_wrappers
from .ltx_distillation.utils import (
    add_noise,
    compute_latent_shapes,
    concat_shot_audios,
    concat_shot_videos,
    decode_benchmark_sample,
    encode_memory_frames_batch,
    save_memory_bank_frames,
    write_benchmark_media,
)
from .ltx_core.model.transformer.model import BlockGPUManager

from .utils import streaming_single_model,streaming_prefetch_model,_full_gpu_ctx,streaming_single_te
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference.yaml"


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class InferenceConfig:
    """Parsed inference configuration from YAML + CLI overrides."""

    def __init__(self, config_path: Path, **cli_overrides):
        cfg = _load_yaml_config(config_path)

        paths_cfg = cfg.get("paths", {})
        video_cfg = cfg.get("video", {})
        denoising_cfg = cfg.get("denoising", {})
        memory_cfg = cfg.get("memory", {})
        audio_cfg = cfg.get("audio_memory", {})
        inference_cfg = cfg.get("inference", {})

        # Paths
        self.checkpoint = str(_resolve_path(paths_cfg.get("checkpoint", "checkpoints/echo-longvideo-release.safetensors"), REPO_ROOT))
        self.gemma_path = str(_resolve_path(paths_cfg.get("gemma_path", "checkpoints/gemma-3-12b"), REPO_ROOT))
        self.prompts_dir = str(_resolve_path(paths_cfg.get("prompts_dir", "prompts"), REPO_ROOT))
        self.prompts_glob = paths_cfg.get("prompts_glob", "*.json")
        self.output_root = str(_resolve_path(paths_cfg.get("output_root", "inference_result/dmd"), REPO_ROOT))

        # Video
        self.num_frames = video_cfg.get("num_frames", 241)
        self.video_height = video_cfg.get("height", 736)
        self.video_width = video_cfg.get("width", 1280)
        self.video_fps = video_cfg.get("fps", 25)
        self.seed = video_cfg.get("seed", 12345)

        # Denoising
        self.denoising_steps = denoising_cfg.get("steps", [])
        self.denoising_sigmas = denoising_cfg.get("sigmas", [])

        # Memory
        self.memory_max_size = memory_cfg.get("max_size", 7)
        self.num_fix_frames = memory_cfg.get("num_fix_frames", 3)
        self.memory_downscale_factor = memory_cfg.get("downscale_factor", 1)
        self.memory_position_mode = memory_cfg.get("position_mode", "reference")
        self.memory_lora_strength = memory_cfg.get("lora_strength", 1.0)
        self.memory_lora_generator = memory_cfg.get("lora_generator", True)
        self.memory_lora_path = memory_cfg.get("lora_path", "") or None
        self.save_mode = memory_cfg.get("save_mode", "random_every_shot_frame")
        self.video_memory_frame_selection_mode = memory_cfg.get("frame_selection_mode", "center")
        self.video_memory_clip_num_frames = memory_cfg.get("clip_num_frames", 9)

        # Audio memory
        self.enable_audio_memory = audio_cfg.get("enable", True)
        self.audio_memory_window_size = audio_cfg.get("window_size", 96)
        self.audio_memory_window_selection_mode = audio_cfg.get("window_selection_mode", "max_response")
        self.audio_memory_sample_rate = audio_cfg.get("sample_rate", 16000)
        self.audio_memory_mel_bins = audio_cfg.get("mel_bins", 128)
        self.audio_memory_mel_hop_length = audio_cfg.get("mel_hop_length", 160)
        self.audio_memory_n_fft = audio_cfg.get("n_fft", 1024)
        self.audio_memory_downsample_factor = audio_cfg.get("downsample_factor", 4)
        self.audio_memory_is_causal = audio_cfg.get("is_causal", True)

        # Inference
        self.device = inference_cfg.get("device", "cuda")
        self.dtype = inference_cfg.get("dtype", "bfloat16")
        self.v2a_grad_scale = inference_cfg.get("v2a_grad_scale", 2.0)
        self.vae_path=""
        self.audio_vae_path=""

        # Misc
        self.prompt_max_chars = None
        self.shot_num_frames = None  # 新增：每个镜头的独立帧数列表

        # Apply CLI overrides
        for key, value in cli_overrides.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def load_joyai_te(gemma_path,connector_path,gemma_root,device, dtype=torch.bfloat16):
    print(f"[Stage 1] Loading text encoder...")
    text_encoder = create_text_encoder_wrapper(
        checkpoint_path=connector_path,
        gemma_root=gemma_root,
        device=device,
        dtype=dtype,
        gemma_path=gemma_path,
    )
    text_encoder.eval()
    return text_encoder


def infer_joyai_text(text_encoder, prompt_files,device):
    if text_encoder.prefetch_count is None:
        text_encoder.text_encoder.to(device)
    cached: dict[Path, list[dict[str, Any]]] = {}

    for prompts_file in prompt_files:
        prompts_file = Path(prompts_file)
        prompts = load_multishot_prompts(prompts_file, prompt_max_chars=None)
        
        if not prompts:
            print(f"[Stage 1] Skipping empty prompts file: {prompts_file}", flush=True)
            cached[prompts_file] = []
            continue
        print(f"[Stage 1] Encoding {len(prompts)} prompts from {prompts_file.name}", flush=True)
        file_conds: list[dict[str, Any]] = []
        for prompt in prompts:
            cond = text_encoder([prompt])
            file_conds.append(
                {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in cond.items()}
            )
            del cond
        cached[prompts_file] = file_conds
    if text_encoder.prefetch_count is None:
        text_encoder.text_encoder.to("cpu")
    del text_encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Stage 1] Text encoder released.")
    return cached


def load_joyai_engine(args):
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cli_overrides = {}
    for key in ["seed", "num_frames", "video_height", "video_width", "video_fps",
                "v2a_grad_scale", "memory_max_size", "num_fix_frames", "enable_audio_memory"]:
        val = getattr(args, key, None)
        if val is not None:
            cli_overrides[key] = val
    if args.prompts_dir:
        cli_overrides["prompts_dir"] = str(Path(args.prompts_dir).expanduser().resolve())
    if args.prompts_glob:
        cli_overrides["prompts_glob"] = args.prompts_glob
    if args.output_root:
        cli_overrides["output_root"] = str(Path(args.output_root).expanduser().resolve())
    if args.audio_vae_path:
        cli_overrides["audio_vae_path"] = args.audio_vae_path
    if args.vae_path:
        cli_overrides["vae_path"] = args.vae_path
    if args.checkpoint:
        cli_overrides["checkpoint"] = args.checkpoint

    cfg = InferenceConfig(config_path, **cli_overrides)

    engine = InferenceEngine(cfg)
    
    # Stage 2: now load the generator + VAEs.
    engine.load_generator()
    return engine



def infer_joyai_video(engine, cached,cli_overrides):
    print(f"[Stage 3] Infer video...")
    #cached_per_file = engine.encode_all_prompts(prompt_files)
    cfg= engine.cfg
    for key, value in cli_overrides.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)    
    
    # Stage 3: run inference for each file using the pre-encoded prompts.
    output_root = Path(engine.cfg.output_root) / "outputs"
    # #for prompts_file in prompt_files:
    # cached = cached_per_file.get(prompts_file, [])
    # if not cached:
    #     continue
    #prompt_name = prompts_file.stem
    final_video=[]
    final_audio=[]
    sample_rate=48000
    for prompts_file,cached in cached.items():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = output_root / f"inference_{timestamp}"
        print("cached video and files save to :",run_output_dir)
        video,audio=engine.run_prompt_file(prompts_file, run_output_dir, cached)
        final_video.append(video)
        final_audio.append(audio["waveform"])
        sample_rate=audio["sample_rate"]

    print(f"[Inference] All {len(cached)} prompt file(s) processed.", flush=True)
    print(f"{final_audio[0].shape}") 
    print(f"{final_video[0].shape}")
    return  torch.cat(final_video,dim=0), {"waveform": torch.cat(final_audio,dim=-1), "sample_rate": sample_rate}



class InferenceEngine:
    """Two-stage inference engine: encode all prompts first, then load generator.

    This avoids holding the text encoder (~24GB) and the video generator in
    memory at the same time. Stage 1 loads only the text encoder, encodes every
    prompt, then completely releases it. Stage 2 loads generator + VAEs.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

        # checkpoint = Path(cfg.checkpoint).expanduser().resolve()
        # gemma_path = Path(cfg.gemma_path).expanduser().resolve()
        # if not checkpoint.exists():
        #     raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        # if not gemma_path.exists():
        #     raise FileNotFoundError(f"Gemma path not found: {gemma_path}")
        self._checkpoint = cfg.checkpoint
        self._gemma_path = ""
        self.prefetch_count=None
        self.enable_tiles=False
        self.streaming_mode="fast"
        # Stage-2 modules — populated by load_generator().
        self.generator = None
        self.video_vae = None
        self.audio_vae = None
        self.base_pipeline = None
        self.memory_pipeline = None
        self.audio_sample_rate: int | None = None
        self.tile_size_in_frames=24
        self.tile_size_in_pixels=512

    # ------------------------------------------------------------------
    # Stage 1: encode prompts, then free text encoder
    # ------------------------------------------------------------------

    def encode_all_prompts(
        self, prompt_files: list[Path]
    ) -> dict[Path, list[dict[str, Any]]]:
        """Load text encoder, encode every prompt across all files, free encoder.

        Returns: {prompt_file: [cond_dict_on_cpu, ...]}
        """
        print(f"[Stage 1] Loading text encoder...", flush=True)
        text_encoder = create_text_encoder_wrapper(
            checkpoint_path=str(self._checkpoint),
            gemma_path=str(self._gemma_path),
            device=self.device,
            dtype=self.dtype,
        )
        text_encoder.eval()

        cached: dict[Path, list[dict[str, Any]]] = {}
        for prompts_file in prompt_files:
            prompts = load_multishot_prompts(prompts_file, prompt_max_chars=self.cfg.prompt_max_chars)
            if not prompts:
                print(f"[Stage 1] Skipping empty prompts file: {prompts_file}", flush=True)
                cached[prompts_file] = []
                continue
            print(f"[Stage 1] Encoding {len(prompts)} prompts from {prompts_file.name}", flush=True)
            file_conds: list[dict[str, Any]] = []
            for prompt in prompts:
                cond = text_encoder([prompt])
                file_conds.append(
                    {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in cond.items()}
                )
                del cond
            cached[prompts_file] = file_conds

        # Fully release the text encoder (GPU + CPU).
        del text_encoder
        import gc
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[Stage 1] Text encoder released.", flush=True)
        return cached

    # ------------------------------------------------------------------
    # Stage 2: load generator + VAEs
    # ------------------------------------------------------------------

    def _model_ctx(self,model,prefetch_count: int | None,) :
        if prefetch_count is not None :
            if self.streaming_mode=="fast":
                return streaming_single_te(
                    model,
                    layers_attr="model.velocity_model.transformer_blocks",
                    target_device=torch.device("cuda"),
                )
            elif self.streaming_mode=="slow":
                    return streaming_single_model(
                        model,
                        layers_attr="model.velocity_model.transformer_blocks",
                        target_device=torch.device("cuda"),
                    )
            elif self.streaming_mode=="prefetch":
                return streaming_prefetch_model(
                    model,
                    layers_attr="model.velocity_model.transformer_blocks",
                    target_device=torch.device("cuda"),
                    prefetch_count=prefetch_count,
                )
            else:
                gpu_manager=BlockGPUManager(block_group_size=prefetch_count)
                gpu_manager.setup_for_inference(model.model.velocity_model)
                model.gpu_manager=gpu_manager
                return _full_gpu_ctx(model)
        
        return _full_gpu_ctx(model)
    
    def load_generator(self) -> None:
        cfg = self.cfg
        print(f"[Stage 2] Loading generator + VAEs from {self._checkpoint}", flush=True)

        loras: tuple[LoraPathStrengthAndSDOps, ...] = ()
        if cfg.memory_lora_path and cfg.memory_lora_generator:
            loras = (
                LoraPathStrengthAndSDOps(
                    str(Path(cfg.memory_lora_path).expanduser()),
                    float(cfg.memory_lora_strength),
                    LTXV_LORA_COMFY_RENAMING_MAP,
                ),
            )
        
        self.generator = create_ltx2_wrapper(
            checkpoint_path=self._checkpoint,
            gemma_path=self._gemma_path,
            device=torch.device("cpu"),
            dtype=self.dtype,
            video_height=int(cfg.video_height),
            video_width=int(cfg.video_width),
            loras=loras,
        )
       
        self.generator.eval()

        # Load VAEs to CPU; we hot-swap encoder/decoders per phase to avoid
        # holding ~30GB generator and VAE decoders on GPU at the same time.
       
        self.video_vae, self.audio_vae = create_vae_wrappers(
            vae_path=self.cfg.vae_path,
            audio_vae_path=self.cfg.audio_vae_path,
            device=torch.device("cpu"),
            dtype=self.dtype,
            with_video_encoder=True,
            with_audio_encoder=True,
            decoder_device=torch.device("cpu"),
        )

        
        self.video_vae.eval()
        self.audio_vae.eval()

        self.generator.denoising_sigmas = torch.tensor(list(cfg.denoising_sigmas), device=self.device, dtype=torch.float32)
        self.generator.memory_downscale_factor=int(cfg.memory_downscale_factor)


        self.audio_sample_rate = self.audio_vae.get_output_sample_rate() or 24000
        print(f"[Stage 2] Generator + VAEs ready.", flush=True)

    # ------------------------------------------------------------------
    # Module hot-swap helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _move(module, target_device) -> None:
        if module is None:
            return
        module.to(target_device)

    def _empty(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _stage_for_denoise(self,) -> None:
        """Generator on GPU; all VAE pieces on CPU."""
        self._move(self.video_vae.encoder, "cpu")
        self._move(self.video_vae.decoder, "cpu")
        self._move(self.audio_vae.encoder, "cpu")
        self._move(self.audio_vae.decoder, "cpu")
        self._move(self.audio_vae.vocoder, "cpu")
        if self.prefetch_count is None:
            self._move(self.generator, self.device)
        self._empty()

    def _stage_for_video_encode(self) -> None:
        """Add video VAE encoder onto GPU alongside the generator (brief use)."""
        self._move(self.video_vae.encoder, self.device)

    def _stage_after_video_encode(self) -> None:
        self._move(self.video_vae.encoder, "cpu")
        self._empty()

    def _stage_for_decode(self) -> None:
        """Generator off GPU; VAE decoders + vocoder on GPU."""
        if self.prefetch_count is None:
            self._move(self.generator, "cpu")
        self._empty()
        self._move(self.video_vae.decoder, self.device)
        self._move(self.audio_vae.decoder, self.device)
        self._move(self.audio_vae.vocoder, self.device)

    def run_prompt_file(
        self,
        prompts_file: Path,
        output_dir: Path,
        cached_conds: list[dict[str, Any]],

    ) -> None:
        """Run multishot inference for a single prompt file using pre-encoded prompts."""
        if self.generator is None:
            raise RuntimeError("call load_generator() before run_prompt_file()")
        cfg = self.cfg
        device = self.device
        dtype = self.dtype

        prompts = load_multishot_prompts(prompts_file, prompt_max_chars=cfg.prompt_max_chars)
        if not prompts:
            print(f"[Engine] No prompts found in {prompts_file}, skipping.", flush=True)
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Engine] Processing {prompts_file.name}: {len(prompts)} shots", flush=True)

        if len(cached_conds) != len(prompts):
            raise ValueError(
                f"cached_conds length ({len(cached_conds)}) does not match prompts count ({len(prompts)})"
            )


        if hasattr(self.generator, 'update_resolution'):
            self.generator.update_resolution(
                video_height=int(cfg.video_height),
                video_width=int(cfg.video_width),
            )

        """
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=int(cfg.num_frames),
            video_height=int(cfg.video_height),
            video_width=int(cfg.video_width),
            batch_size=1,
            video_fps=float(cfg.video_fps),
        )
        """

        memory_bank = PairedAudioVideoMemoryBank(
            max_size=int(cfg.memory_max_size),
            save_mode=str(cfg.save_mode),
            num_fix_frames=int(cfg.num_fix_frames),
        )

        shot_paths: list[Path] = []
        shot_audios: list[torch.Tensor] = []
        shot_videos=[]
        metadata: dict[str, Any] = {
            "checkpoint": cfg.checkpoint,
            "prompts_file": str(prompts_file),
            "output_dir": str(output_dir),
            "denoising_steps": [int(x) for x in cfg.denoising_steps],
            "denoising_sigmas": [float(x) for x in cfg.denoising_sigmas],
            "num_prompts": len(prompts),
            "save_mode": cfg.save_mode,
            "memory_max_size": cfg.memory_max_size,
            "num_fix_frames": cfg.num_fix_frames,
            "enable_audio_memory": cfg.enable_audio_memory,
            "shots": [],
        }

        run_started = time.perf_counter()
        shot_durations: list[dict[str, float]] = []
        #print(self.prefetch_count)
        with self._model_ctx(self.generator,self.prefetch_count) as self.generator:
            for shot_idx, prompt in enumerate(prompts):
                shot_started = time.perf_counter()

                # ========== 优化：计算当前 shot 的 num_frames，支持回退到全局 num_frames ==========
                # 1. 如果配置了 shot_num_frames 列表，且当前 shot_idx 在列表范围内，则使用独立帧数
                # 2. 否则（未配置、或列表长度小于 prompt 数量），回退使用全局的 cfg.num_frames
                if cfg.shot_num_frames and shot_idx < len(cfg.shot_num_frames):
                    current_num_frames = cfg.shot_num_frames[shot_idx]
                else:
                    current_num_frames = int(cfg.num_frames)
                
                # 为当前 shot 计算独立的 shape
                video_shape, audio_shape = compute_latent_shapes(
                    num_frames=current_num_frames,
                    video_height=int(cfg.video_height),
                    video_width=int(cfg.video_width),
                    batch_size=1,
                    video_fps=float(cfg.video_fps),
                )
                print(f"[Engine] Shot {shot_idx + 1}/{len(prompts)} using num_frames={current_num_frames}", flush=True)
                # ==============================================================================

                conditional_dict = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in cached_conds[shot_idx].items()
                }
                prompt_seed = int(cfg.seed) + shot_idx
                memory_size_before = len(memory_bank)

                print(
                    f"[Engine] shot={shot_idx + 1}/{len(prompts)} "
                    f"memory_size_before={memory_size_before} seed={prompt_seed}",
                    flush=True,
                )

                memory_video = None
                memory_audio_kwargs: dict[str, Any] = {}

                # Phase A: denoising — generator on GPU, decoders on CPU.
                self._stage_for_denoise()

                denoise_started = time.perf_counter()
                
                with torch.random.fork_rng(devices=[device]):
                    torch.manual_seed(prompt_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed(prompt_seed)
                    #video_latent=torch.load("D:\\Downloads\\joy_echo_lt_cond.pt")
                    if len(memory_bank) > 0:
                        # Briefly bring video encoder onto GPU.
                        self._stage_for_video_encode()
                        memory_video = encode_memory_frames_batch(
                            video_vae=self.video_vae,
                            batch_memory_frames=[memory_bank.get_memory_frames()],
                            target_h=int(cfg.video_height),
                            target_w=int(cfg.video_width),
                            device=device,
                            dtype=dtype,
                            
                        )
                        self._stage_after_video_encode()

                        memory_audio_kwargs = build_paired_audio_memory_kwargs(
                            memory_bank,
                            enable_audio_memory=bool(cfg.enable_audio_memory),
                            v2a_grad_scale=float(cfg.v2a_grad_scale),
                            memory_position_mode=str(cfg.memory_position_mode),
                        )

                        video_latent, audio_latent = self.generator.generate_BidirectionalMemory(
                            video_shape=tuple(video_shape),
                            audio_shape=tuple(audio_shape),
                            conditional_dict=conditional_dict,
                            memory_video=memory_video,
                            seed=prompt_seed,
                            **memory_audio_kwargs,
                        )
                    else:
                        video_latent, audio_latent = self.generator.generate_BidirectionalAV(
                            video_shape=tuple(video_shape),
                            audio_shape=tuple(audio_shape),
                            conditional_dict=conditional_dict,
                            seed=prompt_seed,
                        )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                denoise_elapsed = time.perf_counter() - denoise_started

                # Release intermediates that are no longer needed before the heavy
                # decoder swap-in.
                del conditional_dict, memory_video, memory_audio_kwargs
                memory_video = None
                memory_audio_kwargs = {}

                # Phase B: decode — generator off GPU, decoders + vocoder on GPU.
                self._stage_for_decode()

                decode_started = time.perf_counter()

                audio_memory_latent = (
                    audio_latent.detach().cpu().contiguous()
                    if (cfg.enable_audio_memory and audio_latent is not None)
                    else None
                )

                video_uint8, audio_waveform = decode_benchmark_sample(
                    self.video_vae, self.audio_vae, video_latent, audio_latent,self.enable_tiles,self.tile_size_in_frames,self.tile_size_in_pixels
                )
               
                if device.type == "cuda":
                    torch.cuda.synchronize()
                decode_elapsed = time.perf_counter() - decode_started
                memory_frames_for_bank = video_uint8_to_pil_frames(video_uint8)
                shot_videos.append(video_uint8.cpu().float() / 255.0)      #[F, H, W, C]
                new_memory_metadata: dict[str, Any] = {}
                if audio_memory_latent is not None:
                    new_memory_metadata = memory_bank.save_memory_slot(
                        memory_frames_for_bank,
                        audio_memory_latent,
                        audio_window_size=int(cfg.audio_memory_window_size),
                        video_clip_num_frames=int(cfg.video_memory_clip_num_frames),
                        audio_waveform=audio_waveform,
                        audio_sample_rate=int(cfg.audio_memory_sample_rate),
                        video_fps=float(cfg.video_fps),
                        audio_window_selection_mode=str(cfg.audio_memory_window_selection_mode),
                        video_frame_selection_mode=str(cfg.video_memory_frame_selection_mode),
                        audio_memory_mel_bins=int(cfg.audio_memory_mel_bins),
                        audio_memory_mel_hop_length=int(cfg.audio_memory_mel_hop_length),
                        audio_memory_n_fft=int(cfg.audio_memory_n_fft),
                        audio_memory_downsample_factor=int(cfg.audio_memory_downsample_factor),
                        audio_memory_is_causal=bool(cfg.audio_memory_is_causal),
                    )

                save_memory_bank_frames(
                    memory_bank.get_memory_frames(),
                    output_dir / "memory_bank" / f"shot_{shot_idx:03d}",
                )

                shot_path = output_dir / f"shot_{shot_idx:03d}.mp4"
                write_result = write_benchmark_media(
                    output_path=shot_path,
                    video_uint8=video_uint8,
                    audio_waveform=audio_waveform,
                    fps=int(cfg.video_fps),
                    audio_sr=int(self.audio_sample_rate),
                )
                shot_paths.append(shot_path)
                if audio_waveform is not None:
                    shot_audios.append(audio_waveform.cpu())

                shot_elapsed = time.perf_counter() - shot_started
                timing = {
                    "denoise_sec": round(denoise_elapsed, 3),
                    "decode_sec": round(decode_elapsed, 3),
                    "total_sec": round(shot_elapsed, 3),
                }
                shot_durations.append(timing)

                metadata["shots"].append(
                    {
                        "shot_idx": int(shot_idx),
                        "prompt": prompt,
                        "output_path": str(shot_path),
                        "memory_size_before": int(memory_size_before),
                        "memory_size_after": int(len(memory_bank)),
                        "new_memory_entry": new_memory_metadata,
                        "audio_latent_shape": list(audio_latent.shape) if audio_latent is not None else None,
                        "wrote_audio_in_mp4": bool(write_result["wrote_audio_in_mp4"]),
                        "wrote_sidecar_wav": bool(write_result["wrote_sidecar_wav"]),
                        "audio_stats": write_result["audio_stats"],
                        "memory_entries": memory_bank.get_memory_metadata(),
                        "timing": timing,
                    }
                )

                print(
                    f"[Engine] shot={shot_idx + 1}/{len(prompts)} done "
                    f"denoise={denoise_elapsed:.1f}s decode={decode_elapsed:.1f}s "
                    f"total={shot_elapsed:.1f}s",
                    flush=True,
                )

                del video_latent, audio_latent, video_uint8, audio_waveform
                del audio_memory_latent, memory_frames_for_bank
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            run_elapsed = time.perf_counter() - run_started
            avg_total = sum(t["total_sec"] for t in shot_durations) / max(len(shot_durations), 1)
            avg_denoise = sum(t["denoise_sec"] for t in shot_durations) / max(len(shot_durations), 1)
            avg_decode = sum(t["decode_sec"] for t in shot_durations) / max(len(shot_durations), 1)
            metadata["timing"] = {
                "run_total_sec": round(run_elapsed, 3),
                "avg_shot_total_sec": round(avg_total, 3),
                "avg_denoise_sec": round(avg_denoise, 3),
                "avg_decode_sec": round(avg_decode, 3),
            }
            print(
                f"[Engine] {prompts_file.name} run_total={run_elapsed:.1f}s "
                f"avg_shot={avg_total:.1f}s (denoise={avg_denoise:.1f}s decode={avg_decode:.1f}s)",
                flush=True,
            )

            combined_path = output_dir / "combined_shots.mp4"
            concat_shot_videos(shot_paths, combined_path)
            combined_audio = concat_shot_audios(shot_audios)
            combined_audio_path = None
            # if combined_audio is not None:
            #     combined_audio_path = output_dir / "combined_shots.wav"
            #     torchaudio.save(str(combined_audio_path), combined_audio, sample_rate=int(self.audio_sample_rate))
            if combined_audio is not None:
                combined_audio_path = output_dir / "combined_shots.wav"
                # soundfile 期望输入形状为 [Frames, Channels]，而 torchaudio 输出为 [Channels, Frames]
                audio_np = combined_audio.cpu().numpy()
                if audio_np.ndim == 2:
                    audio_np = audio_np.T  # [C, T] -> [T, C]
                elif audio_np.ndim == 1:
                    pass # 单声道直接保存
                sf.write(str(combined_audio_path), audio_np, int(self.audio_sample_rate))
            metadata["combined_path"] = str(combined_path)
            metadata["combined_audio_path"] = str(combined_audio_path) if combined_audio_path else None
            metadata["combined_audio_stats"] = audio_waveform_stats(combined_audio)
            metadata_path = output_dir / "run_metadata.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"[Engine] Done: {prompts_file.name} -> {combined_path}", flush=True)
        # print(shot_videos[0].shape) #torch.Size([121, 512, 768, 3])
        # print(combined_audio.shape) #torch.Size([2, 230880])
        # print(self.audio_sample_rate) #48000
        return torch.cat(shot_videos,dim=0), {"waveform": combined_audio.unsqueeze(0), "sample_rate": int(self.audio_sample_rate)}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified inference: load models once, process all prompt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to YAML config file")
    parser.add_argument("--prompts-dir", type=str, default=None, help="Override prompts directory")
    parser.add_argument("--prompts-glob", type=str, default=None, help="Override prompts glob pattern")
    parser.add_argument("--output-root", type=str, default=None, help="Override output root directory")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--video-height", type=int, default=None)
    parser.add_argument("--video-width", type=int, default=None)
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--v2a-grad-scale", type=float, default=None)
    parser.add_argument("--memory-max-size", type=int, default=None)
    parser.add_argument("--num-fix-frames", type=int, default=None)
    parser.add_argument("--enable-audio-memory", type=str_to_bool, default=None)
    return parser.parse_args()


# def main() -> None:
#     args = parse_args()

#     config_path = Path(args.config).expanduser().resolve()
#     if not config_path.exists():
#         raise FileNotFoundError(f"Config file not found: {config_path}")

#     cli_overrides = {}
#     for key in ["seed", "num_frames", "video_height", "video_width", "video_fps",
#                 "v2a_grad_scale", "memory_max_size", "num_fix_frames", "enable_audio_memory"]:
#         val = getattr(args, key, None)
#         if val is not None:
#             cli_overrides[key] = val
#     if args.prompts_dir:
#         cli_overrides["prompts_dir"] = str(Path(args.prompts_dir).expanduser().resolve())
#     if args.prompts_glob:
#         cli_overrides["prompts_glob"] = args.prompts_glob
#     if args.output_root:
#         cli_overrides["output_root"] = str(Path(args.output_root).expanduser().resolve())

#     cfg = InferenceConfig(config_path, **cli_overrides)

#     if len(cfg.denoising_steps) != len(cfg.denoising_sigmas):
#         raise ValueError("denoising steps and sigmas must have the same length")

#     engine = InferenceEngine(cfg)

#     # Discover prompt files
#     prompts_dir = Path(cfg.prompts_dir)
#     prompts_pattern = cfg.prompts_glob
#     if not prompts_pattern.startswith("/"):
#         prompt_files = sorted(prompts_dir.glob(prompts_pattern))
#     else:
#         prompt_files = sorted(Path(p) for p in glob(prompts_pattern))

#     if not prompt_files:
#         raise FileNotFoundError(f"No prompt files matched: {prompts_dir / prompts_pattern}")

#     print(f"[Inference] Found {len(prompt_files)} prompt file(s)", flush=True)

#     # Stage 1: encode all prompts across all files, then release text encoder.
#     cached_per_file = engine.encode_all_prompts(prompt_files)

#     # Stage 2: now load the generator + VAEs.
#     engine.load_generator()

#     # Stage 3: run inference for each file using the pre-encoded prompts.
#     output_root = Path(cfg.output_root) / "outputs"
#     for prompts_file in prompt_files:
#         cached = cached_per_file.get(prompts_file, [])
#         if not cached:
#             continue
#         prompt_name = prompts_file.stem
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         run_output_dir = output_root / prompt_name / f"inference_{timestamp}"
#         engine.run_prompt_file(prompts_file, run_output_dir, cached)

#     print(f"[Inference] All {len(prompt_files)} prompt file(s) processed.", flush=True)


# if __name__ == "__main__":
#     main()
