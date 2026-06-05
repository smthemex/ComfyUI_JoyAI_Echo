"""Layer streaming wrapper for memory-efficient inference.
Keeps most transformer/decoder layers on CPU pinned memory and streams them
to GPU on demand, using a secondary CUDA stream to prefetch upcoming layers
so that data transfer overlaps with compute.
General-purpose: works with any ``nn.Module`` whose forward iterates over a
``nn.ModuleList`` attribute (e.g. ``transformer_blocks``, ``layers``).
Each layer is evicted back to CPU immediately after its forward completes,
and prefetch uses modular indexing so the last layer's prefetch wraps around
to prepare early layers for the next forward pass.
Example
-------
>>> model = build_my_model(device=torch.device("cpu"))
>>> model = LayerStreamingWrapper(
...     model,
...     layers_attr="transformer_blocks",
...     target_device=torch.device("cuda:0"),
...     prefetch_count=2,
... )
>>> out = model(inputs)            # hooks handle layer streaming
>>> model.teardown()               # move everything back to CPU
"""

from __future__ import annotations

import functools
import itertools
import logging
from typing import Any

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

# edit from LayerStreamingWrapper from https://github.com/Lightricks/LTX-2

class _SimpleLayerStore:
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
class SimpleLayerStore:
    """简化版层存储，支持按需加载和立即释放"""
    
    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)
        
        # 保留CPU端的原始参数引用，确保递归收集所有子模块
        self._cpu_params: list[dict[str, torch.Tensor]] = []
        for layer in layers:
            cpu_copy = {}
            # 显式遍历所有子模块，确保深层参数（如 attn2.to_q.weight）被包含
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                cpu_copy[name] = tensor.data.cpu()  # 保留在CPU上
            self._cpu_params.append(cpu_copy)
    
    def load_layer_to_gpu(self, idx: int, layer: nn.Module) -> None:
        """将指定层加载到GPU"""
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in self._cpu_params[idx]:
                # 使用 copy_ 原地更新，确保模块内部绑定的权重被正确转移到 GPU
                param.data.copy_(self._cpu_params[idx][name].to(self.target_device))
                # 记录流，防止显存被提前回收
                param.data.record_stream(torch.cuda.current_stream(self.target_device))

    def unload_layer_from_gpu(self, idx: int, layer: nn.Module) -> None:
        """将指定层从GPU卸载回CPU"""
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if name in self._cpu_params[idx]:
                # 使用 copy_ 原地更新，确保模块内部绑定的权重被正确移回 CPU
                param.data.copy_(self._cpu_params[idx][name])
class SimpleLayerStreamingWrapper_(nn.Module):
    """简化版层流式处理包装器"""
    
    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        active_count: int = 1,  # 同时激活的层数量
    ) -> None:
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        self._active_count = active_count
        self._store = SimpleLayerStore(self._layers, self._target_device)
        
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
            # 加载当前层到GPU（内部已包含record_stream）
            self._store.load_layer_to_gpu(idx, module)
        
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
    
class SimpleLayerStreamingWrapper(nn.Module):
    """简化版层流式处理包装器"""
    
    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        active_count: int = 1,  # 同时激活的层数量
    ) -> None:
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        self._active_count = active_count
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
    

