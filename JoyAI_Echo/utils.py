from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

from contextlib import contextmanager
from .layer_streaming import SimpleLayerStreamingWrapper,LayerStreamingWrapper,SimpleLayerTEWrapper
from collections.abc import Iterator
from typing import TypeVar
import gc
import torch
# from utils import apply_loras_gguf

_M = TypeVar("_M", bound=torch.nn.Module)
T = TypeVar("T")


@contextmanager
def _full_gpu_ctx(model,device=None):
    """Context manager to load the entire model to GPU and release it after use."""
    try:
        if device is not None:
            device = torch.device(device)
            model.to(device)
        yield model
    finally:
        model.to("cpu")


def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# LayerStreamingWrapper from https://github.com/Lightricks/LTX-2

@contextmanager
def streaming_single_model(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
) -> Iterator[_M]:
    """Wrap *model* with :class:`LayerStreamingWrapper`, yield it, then tear down."""
    wrapped = SimpleLayerStreamingWrapper(
        model,
        layers_attr=layers_attr,
        target_device=target_device,
    )
    try:
        yield wrapped  # type: ignore[misc]
    finally:
        wrapped.to("cpu")
        cleanup_memory()
        # Flush the host (pinned) memory cache so that freed pinned pages are
        # returned to the OS.  Without this, sequential streaming models
        # (e.g. text encoder then transformer) exhaust host memory because the
        # CachingHostAllocator keeps freed blocks cached indefinitely.
        torch.cuda.synchronize(device=target_device)
        try:
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
        except Exception:
            print("Host empty cache cleanup failed; ignoring.", exc_info=True)

@contextmanager
def streaming_single_te(
    model: _M,  # 模型参数，类型为_M
    layers_attr: str,  # 属性字符串，用于指定模型中的层
    target_device: torch.device,  # 目标设备，用于指定模型运行在哪个设备上
) -> Iterator[_M]:
    """Wrap *model* with :class:`LayerStreamingWrapper`, yield it, then tear down."""
    wrapped = SimpleLayerTEWrapper(
        model,
        layers_attr=layers_attr,
        target_device=target_device,
    )
    try:
        yield wrapped  # type: ignore[misc]
    finally:
        wrapped.to("cpu")
        cleanup_memory()
        # Flush the host (pinned) memory cache so that freed pinned pages are
        # returned to the OS.  Without this, sequential streaming models
        # (e.g. text encoder then transformer) exhaust host memory because the
        # CachingHostAllocator keeps freed blocks cached indefinitely.
        torch.cuda.synchronize(device=target_device)
        try:
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
        except Exception:
            print("Host empty cache cleanup failed; ignoring.", exc_info=True)


@contextmanager
def streaming_prefetch_model(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
    prefetch_count: int,
) -> Iterator[_M]:
    """Wrap *model* with :class:`LayerStreamingWrapper`, yield it, then tear down."""
    wrapped = LayerStreamingWrapper(
        model,
        layers_attr=layers_attr,
        target_device=target_device,
        prefetch_count=prefetch_count,
    )
    try:
        yield wrapped  # type: ignore[misc]
    finally:
        # if hasattr(wrapped, 'teardown'):
        #     wrapped.teardown()
        wrapped.to("cpu")
        cleanup_memory()
        # Flush the host (pinned) memory cache so that freed pinned pages are
        # returned to the OS.  Without this, sequential streaming models
        # (e.g. text encoder then transformer) exhaust host memory because the
        # CachingHostAllocator keeps freed blocks cached indefinitely.
        torch.cuda.synchronize(device=target_device)
        try:
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
        except Exception:
            print("Host empty cache cleanup failed; ignoring.", exc_info=True)

def set_gguf2meta_model(meta_model,model_state_dict,dtype,device,lora_sd=None):
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import GGUFQuantizer

    g_config = GGUFQuantizationConfig(compute_dtype=dtype or torch.bfloat16)
    hf_quantizer = GGUFQuantizer(quantization_config=g_config)
    hf_quantizer.pre_quantized = True
    if lora_sd is not None:
        try:
            model_state_dict=apply_loras_gguf(model_state_dict, lora_sd)
            print("Applying LoRAs to GGUF model success>")
        except Exception as e:
            print(f"Error applying LoRAs to GGUF model: {e}")
            pass

    hf_quantizer._process_model_before_weight_loading(
        meta_model,
        device_map={"": device} if device else None,
        state_dict=model_state_dict
    )
    from diffusers.models.model_loading_utils import load_model_dict_into_meta
    load_model_dict_into_meta(
        meta_model, 
        model_state_dict, 
        hf_quantizer=hf_quantizer,
        device_map={"": device} if device else None,
        dtype=dtype
    )

    hf_quantizer._process_model_after_weight_loading(meta_model)
    
    del model_state_dict
    gc.collect()
    
    return meta_model.to(dtype=dtype)

def load_gguf_checkpoint_gemma(gguf_checkpoint_path):

    from  diffusers.utils  import is_gguf_available, is_torch_available
    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader
        from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter,dequantize_gguf_tensor
    else:
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    parsed_parameters = {}
 
    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type

        # if the tensor is a torch supported dtype do not use GGUFParameter
        is_gguf_quant = quant_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: https://github.com/huggingface/diffusers"
                )
            )

        weights = torch.from_numpy(tensor.data.copy())
        parsed_parameters[name] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights
    
    del reader
    gc.collect()
    return parsed_parameters


def match_state_dict(meta_model, sd,show_num=10):

    meta_model_keys = set(meta_model.state_dict().keys())   
    state_dict_keys = set(sd.keys())

    matching_keys = meta_model_keys.intersection(state_dict_keys)
    print(f"Matching keys count: {len(matching_keys)}")
    

    extra_keys = state_dict_keys - meta_model_keys
    if extra_keys:
        print(f"Extra keys in state_dict (not in meta_model): {len(extra_keys)}")
        for key in list(extra_keys)[:show_num]: 
            print(f"  - {key}")
    
    missing_keys = meta_model_keys - state_dict_keys
    if missing_keys:
        print(f"Missing keys in state_dict (not in state_dict): {len(missing_keys)}")
        for key in list(missing_keys)[:show_num]:  
            print(f"  - {key}")
    
    print(f"Sample matching keys: {list(matching_keys)[:5]}")

def load_gguf_checkpoint(gguf_checkpoint_path):

    from  diffusers.utils  import is_gguf_available, is_torch_available
    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader
        from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter,dequantize_gguf_tensor
    else:
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    parsed_parameters = {}
 
    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type

        # if the tensor is a torch supported dtype do not use GGUFParameter
        is_gguf_quant = quant_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: https://github.com/huggingface/diffusers"
                )
            )

        weights = torch.from_numpy(tensor.data.copy())
        parsed_parameters[name] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights
        del tensor,weights
    del reader
    gc.collect()
    return parsed_parameters

def apply_loras_gguf(
    model_sd,
    lora_sd,
):
    sd = {}
    for key, weight in model_sd.items():
        if weight is None:
            continue
        device = weight.device
        deltas_dtype =  torch.bfloat16
        deltas = _prepare_deltas(lora_sd, key, deltas_dtype, device)
        if deltas is None:
            sd[key] = weight
        else:
            deltas = deltas.to(dtype=deltas_dtype)
            if  getattr(weight,"quant_type",False):
                try:
                    weight = (dequantize_gguf_tensor(weight).to(dtype=deltas_dtype)) + deltas
                    sd[key] = weight
                except Exception as e:
                    print(f"Error dequantizing GGUF weight for {key}: {e}")
                    sd[key] = weight
            else:
                sd[key] = weight + deltas
            
        del weight,deltas
    del model_sd
    gc.collect()
    return sd

def _prepare_deltas( lora_sd,key: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    deltas = None
    prefix = key[: -len(".weight")]
    key_a = f"{prefix}.lora_down.weight"
    key_b = f"{prefix}.lora_up.weight"
    lora_alpha = f"{prefix}.alpha"
    if key_a  in lora_sd :
        lora_down = lora_sd[key_a].to(device=device)
        lora_up = lora_sd[key_b].to(device=device)
        alpha = float(lora_sd.get(lora_alpha, 1.0))
        rank = lora_down.shape[0]
        scaling_factor = alpha / rank
        deltas = scaling_factor * torch.matmul(lora_up, lora_down).to(device)
        del lora_down, lora_up,alpha
    return deltas


