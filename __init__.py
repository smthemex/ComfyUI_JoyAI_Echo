
from comfy_api.latest import ComfyExtension,io
from typing_extensions import override

from .JoyAI_Echo_node import JoyAI_Echo_SM_Model, JoyAI_Echo_SM_Clip, JoyAI_Echo_SM_Encoder, JoyAI_Echo_SM_KSampler
class  JoyAI_Echo_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            JoyAI_Echo_SM_Model,
            JoyAI_Echo_SM_Clip,
            JoyAI_Echo_SM_Encoder,
            JoyAI_Echo_SM_KSampler,
        ]
async def comfy_entrypoint() -> JoyAI_Echo_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return JoyAI_Echo_SM_Extension()