from __future__ import annotations
import functools
import itertools
import logging
from typing import Any
import torch.nn.functional as F
import torch
from torch import nn

logger = logging.getLogger(__name__)



def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


class SimpleLayerStore_TE:
    """简化版层存储，支持按需加载和立即释放"""
    
    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)
        
        # 保留CPU端的原始参数引用
        self._cpu_params: list[dict[str, torch.Tensor]] = []
        for layer in layers:
            cpu_copy = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                cpu_copy[name] = tensor.data.cpu()  # 保留在CPU上
            self._cpu_params.append(cpu_copy)
    
    def load_layer_to_gpu(self, idx: int, layer: nn.Module) -> None:
        """将指定层加载到GPU"""
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name].to(self.target_device)
    
    def unload_layer_from_gpu(self, idx: int, layer: nn.Module) -> None:
        """将指定层从GPU卸载回CPU"""
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name]  # 恢复为CPU副本


class SimpleLayerTEWrapper(nn.Module):
    """单层流式处理包装器"""
    
    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
    ) -> None:
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device

        self._store = SimpleLayerStore_TE(self._layers, self._target_device)
        
        # 将非层参数移到GPU
        self._move_non_layer_params_to_gpu()
        
        # 注册钩子
        self._register_simple_hooks()
    
    def _move_non_layer_params_to_gpu(self) -> None:
        """移动非层参数到GPU"""
        layer_tensor_ids = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)
    
    def _register_simple_hooks(self) -> None:
        """注册简单的加载/释放钩子"""
        idx_map = {id(layer): idx for idx, layer in enumerate(self._layers)}
        
        def _pre_hook(module: nn.Module, input, *, idx: int):
            # 加载当前层到GPU
            self._store.load_layer_to_gpu(idx, module)
            # 记录流，防止内存被提前回收
            for param in itertools.chain(module.parameters(), module.buffers()):
                param.data.record_stream(torch.cuda.current_stream(self._target_device))
        
        def _post_hook(module: nn.Module, input, output, *, idx: int):
            # 处理完后立即将层移回CPU
            self._store.unload_layer_from_gpu(idx, module)
        
        for layer in self._layers:
            idx = idx_map[id(layer)]
            pre_hook = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            post_hook = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """代理属性访问到原始模型"""
        try:
            # 首先尝试从包装器自身获取属性
            return super().__getattr__(name)
        except AttributeError:
            # 如果失败，则从原始模型获取
            return getattr(self._model, name)

class _SimpleLayerStore:
    """Layer store that preserves original CPU tensors to avoid reallocation."""

    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)
        # 按层存储每个参数的原始 CPU 张量引用
        self._cpu_refs: list[dict[str, torch.Tensor]] = []

    def _ensure_cpu_refs(self, idx: int, layer: nn.Module) -> None:
        """首次加载时保存当前参数的 CPU 引用（如果参数还在 CPU 上）。"""
        if idx < len(self._cpu_refs):
            return  # 已经保存过
        refs = {}
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if param.data.device.type == 'cpu':
                refs[name] = param.data  # 直接保存当前 CPU 张量的引用
            else:
                # 正常情况下不会发生，但安全起见先移到 CPU 再保存引用
                refs[name] = param.data.cpu()
        # 确保 _cpu_refs 长度足够
        while len(self._cpu_refs) <= idx:
            self._cpu_refs.append({})
        self._cpu_refs[idx] = refs

    def load_layer_to_gpu(self, idx: int, layer: nn.Module) -> None:
        # 第一次加载时记录原始 CPU 引用
        self._ensure_cpu_refs(idx, layer)
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            param.data = self._cpu_refs[idx][name].to(self.target_device)

    def unload_layer_from_gpu(self, idx: int, layer: nn.Module) -> None:
        """恢复为最初的 CPU 张量（不分配新内存）。"""
        refs = self._cpu_refs[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if param.data.is_cuda:
                param.data = refs[name]   # 直接指回原始 CPU 张量

class SimpleLayerStreamingWrapper(nn.Module):
    """单层流式处理包装器"""
    
    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
    ) -> None:
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device

        self._store = _SimpleLayerStore(self._layers, self._target_device)
        
        # 将非层参数移到GPU
        self._move_non_layer_params_to_gpu()
        
        # 注册钩子
        self._register_simple_hooks()
    
    def _move_non_layer_params_to_gpu(self) -> None:
        """移动非层参数到GPU"""
        layer_tensor_ids = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)
    
    def _register_simple_hooks(self) -> None:
        """注册简单的加载/释放钩子"""
        idx_map = {id(layer): idx for idx, layer in enumerate(self._layers)}
        
        def _pre_hook(module: nn.Module, input, *, idx: int):
            # 加载当前层到GPU
            self._store.load_layer_to_gpu(idx, module)
            # 记录流，防止内存被提前回收
            compute_stream = torch.cuda.current_stream(self._target_device)
            for param in itertools.chain(module.parameters(), module.buffers()):
                if param.data.is_cuda and param.data.data_ptr() != 0:
                    current_tensor_stream = torch.cuda.current_stream(param.data.device)
                    if current_tensor_stream != compute_stream:
                        param.data.record_stream(compute_stream)
        
        def _post_hook(module: nn.Module, input, output, *, idx: int):
            # 处理完后立即将层移回CPU
            self._store.unload_layer_from_gpu(idx, module)
        
        for layer in self._layers:
            idx = idx_map[id(layer)]
            pre_hook = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            post_hook = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """代理属性访问到原始模型"""
        try:
            # 首先尝试从包装器自身获取属性
            return super().__getattr__(name)
        except AttributeError:
            # 如果失败，则从原始模型获取
            return getattr(self._model, name)


# class _LayerStore:
#     def __init__(self, layers, target_device):
#         self.target_device = target_device
#         self._on_gpu = set()
#         self._cpu_refs: list[dict[str, torch.Tensor]] = []  # 只存普通 CPU 引用

#         for layer in layers:
#             refs = {}
#             for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
#                 refs[name] = param.data  # 原始 CPU 张量
#             self._cpu_refs.append(refs)

#     def mark_on_gpu(self, idx):
#         self._on_gpu.add(idx)

#     def evict_to_cpu(self, idx, layer):
#         if idx not in self._on_gpu:
#             return
#         refs = self._cpu_refs[idx]
#         for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
#             param.data = refs[name]  # 直接指回原始 CPU 张量
#         self._on_gpu.discard(idx)

# class _AsyncPrefetcher:
#     def prefetch(self, idx):
#         if idx in self._store._on_gpu or idx in self._events:
#             return
#         layer = self._layers[idx]

#         # 1. 临时 pin 当前层参数
#         pinned = {}
#         for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
#             pinned[name] = param.data.pin_memory()  # 分配新的 pinned 副本

#         # 2. 在专用 stream 上异步拷贝到 GPU
#         with torch.cuda.stream(self._stream):
#             for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
#                 param.data = pinned[name].to(self._store.target_device, non_blocking=True)
#             event = torch.cuda.Event()
#             event.record(self._stream)
#             self._events[idx] = event

#         # 3. 立即释放 pinned 内存
#         del pinned
#         self._store.mark_on_gpu(idx)


class _LayerStore:
    def __init__(self, layers, target_device):
        self.target_device = target_device
        self.num_layers = len(layers)
        self._on_gpu = set()
        # 按层记录每个参数的原始 CPU 张量引用（不强制 pin_memory，避免 32G 内存翻倍）
        self._cpu_tensors: list[dict[str, torch.Tensor]] = []

        # 初始化时，直接保存原模型 CPU 张量的引用
        for layer in layers:
            layer_refs = {}
            for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                # 确保参数在 CPU 上，并保存引用
                if param.data.device.type != 'cpu':
                    param.data = param.data.cpu()
                layer_refs[name] = param.data
            self._cpu_tensors.append(layer_refs)

    def is_on_gpu(self, idx):
        return idx in self._on_gpu

    def move_to_gpu(self, idx, layer, non_blocking=False):
        if idx in self._on_gpu:
            return
        cpu_tensors = self._cpu_tensors[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            # 从 CPU 传输到 GPU
            param.data = cpu_tensors[name].to(self.target_device, non_blocking=non_blocking)
        self._on_gpu.add(idx)

    def evict_to_cpu(self, idx, layer):
        if idx not in self._on_gpu:
            return
        cpu_tensors = self._cpu_tensors[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if param.data.is_cuda:
                # 关键修复：必须将 GPU 数据同步拷贝回 CPU 副本
                # 这样才能强制触发 D2H 传输，并在完成后让 Caching Allocator 物理释放 GPU 显存
                cpu_tensors[name].copy_(param.data, non_blocking=False)
                # 将 param.data 指回 CPU，确保原 GPU 张量引用计数归零
                param.data = cpu_tensors[name]
        self._on_gpu.discard(idx)

    def cleanup(self):
        self._on_gpu.clear()
        self._cpu_tensors.clear()



class _AsyncPrefetcher:
    """Issues H2D transfers on a dedicated CUDA stream.
    Uses per-layer CUDA events so that the compute stream only waits for the
    specific layer it needs, not all pending transfers.
    """

    def __init__(self, store: _LayerStore, layers: nn.ModuleList) -> None:
        self._store = store
        self._layers = layers
        self._stream = torch.cuda.Stream(device=store.target_device)
        self._events: dict[int, torch.cuda.Event] = {}

    def prefetch(self, idx: int) -> None:
        """Begin async transfer of layer *idx* to GPU (no-op if already there)."""
        if self._store.is_on_gpu(idx) or idx in self._events:
            return
        with torch.cuda.stream(self._stream):
            self._store.move_to_gpu(idx, self._layers[idx], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._stream)
            self._events[idx] = event

    def wait(self, idx: int) -> None:
        """Block the compute stream until layer *idx* transfer is complete."""
        event = self._events.pop(idx, None)
        if event is not None:
            torch.cuda.current_stream(self._store.target_device).wait_event(event)

    def cleanup(self) -> None:
        """Drain pending work and release CUDA stream/event resources."""
        self._events.clear()
        self._stream = None
        self._layers = None
        self._store = None




class LayerStreamingWrapper(nn.Module):
    """Wraps a model to stream its sequential layers between CPU and GPU.
    Each layer is evicted immediately after its forward completes, and
    prefetch wraps around using modular indexing so the end of one forward
    pass prepares early layers for the next.
    Parameters
    ----------
    model:
        The model to wrap, with all parameters on **CPU**.
    layers_attr:
        Dotted attribute path to the ``nn.ModuleList`` of sequential layers
        (e.g. ``"transformer_blocks"`` or ``"model.language_model.layers"``).
    target_device:
        The GPU device to use for compute.
    prefetch_count:
        How many layers ahead to prefetch.  The maximum number of layers on
        GPU at once is ``1 + prefetch_count``.  Must be >= 1.
    """

    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        prefetch_count: int = 2,
    ) -> None:
        if prefetch_count < 1:
            raise ValueError("prefetch_count must be >= 1")
        super().__init__()
        # Store the wrapped model as a submodule so parameters are discoverable.
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        # Clamp: no point prefetching more than num_layers - 1 (the rest are evicted).
        self._prefetch_count = min(prefetch_count, len(self._layers) - 1)
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        self._setup()

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------


    def _setup(self) -> None:
        # 1. Build the pinned CPU store (copies all layer tensors to pinned memory).
        self._store = _LayerStore(self._layers, self._target_device)

        # 2. Move all NON-layer params/buffers to GPU.
        layer_tensor_ids: set[int] = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)

        # 3. Pre-load the first (1 + prefetch_count) layers synchronously.
        for idx in range(min(self._prefetch_count + 1, len(self._layers))):
            self._store.move_to_gpu(idx, self._layers[idx])

        # 4. Create the async prefetcher and register hooks.
        self._prefetcher = _AsyncPrefetcher(self._store, self._layers)
        self._register_hooks()


    def _register_hooks(self) -> None:
        idx_map: dict[int, int] = {id(layer): idx for idx, layer in enumerate(self._layers)}
        num_layers = len(self._layers)

        def _pre_hook(
            module: nn.Module,
            _args: Any,  # noqa: ANN401
            *,
            idx: int,
        ) -> None:
            # Wait only for THIS layer's H2D transfer (not all pending ones).
            self._prefetcher.wait(idx)
            if not self._store.is_on_gpu(idx):
                self._store.move_to_gpu(idx, module)

            # Record that the compute stream will read these weight tensors.
            # They were allocated on the prefetch stream, so without this the
            # caching allocator would allow the prefetch stream to reuse their
            # memory immediately after eviction — even if the compute kernel
            # that reads them hasn't finished yet.
            compute_stream = torch.cuda.current_stream(self._target_device)
            for param in itertools.chain(module.parameters(), module.buffers()):
                param.data.record_stream(compute_stream)

            # Kick off prefetch for upcoming layers (wraps around for next pass).
            for offset in range(1, self._prefetch_count + 1):
                self._prefetcher.prefetch((idx + offset) % num_layers)

        def _post_hook(
            module: nn.Module,
            _args: Any,  # noqa: ANN401
            _output: Any,  # noqa: ANN401
            *,
            idx: int,
        ) -> None:
            # Evict this layer immediately — its computation is done.
            self._store.evict_to_cpu(idx, module)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            h1 = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            h2 = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
            self._hooks.extend([h1, h2])

    def teardown(self) -> None:
        """Remove hooks, release memory, and move parameters back to CPU."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        torch.cuda.synchronize(device=self._target_device)
        if self._prefetcher is not None:
            self._prefetcher.cleanup()
            self._prefetcher = None

        for idx, layer in enumerate(self._layers):
            self._store.evict_to_cpu(idx, layer)

        for p in self._model.parameters():
            p.data = p.data.to("cpu")
        for b in self._model.buffers():
            b.data = b.data.to("cpu")

        self._store.cleanup()



    # ------------------------------------------------------------------
    # Forward and attribute delegation
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Proxy attribute access to the wrapped model.
        This allows calling methods like ``encode()`` on a wrapped
        GemmaTextEncoder without the caller needing to know about the wrapper.
        ``nn.Module.__getattr__`` is only called when normal attribute lookup
        fails, so ``_model``, ``_store``, etc. are found first via ``__dict__``.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
        
