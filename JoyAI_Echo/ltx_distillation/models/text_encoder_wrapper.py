"""
Gemma Text Encoder Wrapper for DMD distillation.

Provides a simple interface for text encoding without prompt enhancement.
Just pure text -> context embedding conversion.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

from ...ltx_core.loader.registry import Registry
from ...utils import streaming_single_te,streaming_prefetch_model,_full_gpu_ctx

class GemmaTextEncoderWrapper(nn.Module):
    """
    Wrapper for Gemma text encoder to provide DMD-compatible interface.

    This wrapper:
    - Takes raw text prompts (no enhancement needed)
    - Returns conditional_dict with video_context and audio_context
    - Handles batched encoding
    """

    def __init__(
        self,
        text_encoder,
        embeddings_processor,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            text_encoder: GemmaTextEncoder instance
            embeddings_processor: EmbeddingsProcessor instance
            device: Target device
            dtype: Model dtype
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.embeddings_processor = embeddings_processor
        self.device = device
        self.dtype = dtype
        self.prefetch_count=None
        self.encode_count=0
        self.enable_streaming=False


    def _model_ctx(self,model,prefetch_count: int | None,) :
        if prefetch_count is not None:
            if not self.enable_streaming:
                return streaming_single_te(
                    model,
                    layers_attr="model.model.language_model.layers",
                    target_device=torch.device("cuda"),
                )
            else:
                return streaming_prefetch_model(
                    model,
                    layers_attr="model.model.language_model.layers",
                    target_device=torch.device("cuda"),
                    prefetch_count=prefetch_count,
                )
        return _full_gpu_ctx(model)


    @torch.no_grad()
    def forward(
        self,
        text_prompts: List[str],
        padding_side: str = "left",
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Encode text prompts to conditioning embeddings.

        Args:
            text_prompts: List of text prompts (already processed, no enhancement)
            padding_side: Padding side for tokenizer

        Returns:
            Dictionary containing:
                - video_context: [B, seq_len, dim] video conditioning
                - audio_context: [B, seq_len, dim] audio conditioning
                - attention_mask: [B, seq_len] attention mask
        """
        batch_size = len(text_prompts)
        print("encode prompt:",self.encode_count)
        self.encode_count+=1
        # Encode each prompt
        video_contexts = []
        audio_contexts = []
        attention_masks = []
        with self._model_ctx(self.text_encoder, self.prefetch_count) as self.text_encoder:
            for prompt in text_prompts:
                # 1) Run Gemma LLM to get raw hidden states + attention mask
                hidden_states, attn_mask = self.text_encoder.encode(prompt, padding_side=padding_side)
                # 2) Process hidden states to obtain final embeddings
                output = self.embeddings_processor.process_hidden_states(
                    hidden_states, attn_mask, padding_side=padding_side
                )

                video_contexts.append(output.video_encoding)
                audio_contexts.append(output.audio_encoding)
                attention_masks.append(output.attention_mask)

            # Stack batch
            video_context = torch.cat(video_contexts, dim=0) if len(video_contexts) > 0 else None
            # Handle optional audio connector (may be None depending on config)
            if any(ac is None for ac in audio_contexts):
                audio_context = None
            else:
                audio_context = torch.cat(audio_contexts, dim=0)
            attention_mask = torch.cat(attention_masks, dim=0) if len(attention_masks) > 0 else None

            return {
                "video_context": video_context,
                "audio_context": audio_context,
                "attention_mask": attention_mask,
            }

    def encode_batch(
        self,
        text_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Alias for forward() with default padding."""
        return self.forward(text_prompts)


def create_text_encoder_wrapper(
    checkpoint_path: str,
    gemma_root: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    registry: Registry | None = None,
    gemma_path: str | None = None,
) -> GemmaTextEncoderWrapper:
    """
    Factory function to create GemmaTextEncoderWrapper from checkpoint.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint
        gemma_path: Path to Gemma text encoder
        device: Target device
        dtype: Model dtype

    Returns:
        Configured GemmaTextEncoderWrapper
    """
    from ...ltx_pipelines.utils.model_ledger import ModelLedger

    # Load to CPU first to avoid safetensors device issues
    model_ledger = ModelLedger(
        dtype=torch.bfloat16,
        device="cpu",
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_root,
        loras=[],
        quantization=None,
        gguf_dit=True ,
        load_model="clip",
        clip_path=gemma_path,
    )


    # ledger = ModelLedger(
    #     dtype=dtype,
    #     device=torch.device("cpu"),
    #     checkpoint_path=checkpoint_path,
    #     gemma_root_path=gemma_path,
    #     registry=registry,
    # )

    # text_encoder = ledger.text_encoder().to(device=device, dtype=dtype)
    # embeddings_processor = ledger.gemma_embeddings_processor().to(device=device, dtype=dtype)

    wrapper = GemmaTextEncoderWrapper(
        text_encoder=model_ledger.text_encoder(),
        embeddings_processor=model_ledger.gemma_embeddings_processor(),
        device=device,
        dtype=dtype,
    )
    del model_ledger
    return wrapper


