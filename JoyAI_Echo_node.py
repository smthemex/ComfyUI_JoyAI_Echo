 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import folder_paths
from comfy_api.latest import  io
import nodes
from pathlib import PureWindowsPath
from .JoyAI_Echo.inference import load_joyai_te,infer_joyai_text,load_joyai_engine,infer_joyai_video
from .node_utils import clear_comfyui_cache,create_temp_json
MAX_SEED = np.iinfo(np.int32).max

node_joyai_echo_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir


class JoyAI_Echo_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoyAI_Echo_SM_Model",
            display_name="JoyAI_Echo_SM_Model",
            category="JoyAI_Echo_SM",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
                io.Combo.Input("audio_vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
                
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls,dit,gguf,vae,audio_vae) -> io.NodeOutput:
        clear_comfyui_cache()
        dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None 
        vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
        audio_vae_path=folder_paths.get_full_path("vae", audio_vae) if audio_vae != "none" else None
        import argparse
        args = argparse.Namespace(
            config=os.path.join(node_joyai_echo_path, "JoyAI_Echo/configs/inference.yaml"),
            device="cuda",
            dtype="bfloat16",
            checkpoint=dit_path or gguf_path,
            output_root=folder_paths.get_output_directory(),
            prompts_dir=folder_paths.get_output_directory(),
            prompts_glob="*.json",
            vae_path=vae_path,
            audio_vae_path=audio_vae_path,
        )
        model= load_joyai_engine(args)
        return io.NodeOutput(model)

class JoyAI_Echo_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoyAI_Echo_SM_KSampler",
            display_name="JoyAI_Echo_SM_KSampler",
            category="JoyAI_Echo_SM",
            inputs=[
                io.Model.Input("model"),     
                io.Int.Input("width", default=768, min=256, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=512, min=256, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Int.Input("num_frames", default=121, min=16, max=MAX_SEED,step=1,display_mode=io.NumberDisplay.number),
                io.Float.Input("frame_rate", default=24.0, min=8.0, max=120.0,step=1.0,display_mode=io.NumberDisplay.number),
                io.Int.Input("prefetch_count", default=1, min=0, max=48,step=1,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("enable_tile", default=False),
                io.Boolean.Input("enable_streaming", default=False),
                io.Conditioning.Input("te_cond",optional=True),
            ], 
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
            ],
        )
    @classmethod
    def execute(cls, model,width,height,seed,num_frames,frame_rate,prefetch_count,enable_tile,enable_streaming,te_cond=None) -> io.NodeOutput:
        clear_comfyui_cache()
        if te_cond is None:
            if not os.path.exists(os.path.join(folder_paths.get_output_directory(),"joy_echo_te_cond.pt")):
                raise Exception("te_cond is None or comfyUI outpu dont exist joy_echo_te_cond.pt  ")
            te_cond = torch.load(os.path.join(folder_paths.get_output_directory(),"joy_echo_te_cond.pt"),weights_only=False)
        model.prefetch_count=prefetch_count if prefetch_count > 0 else None
        model.enable_tile=enable_tile
        model.enable_streaming=enable_streaming
        cli_overrides = {
            "video_width": width,
            "video_height": height,
            # "steps": steps,
            "seed": seed,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
        }

        images,audio=infer_joyai_video(model, te_cond,cli_overrides)

        return io.NodeOutput(images,audio)


class JoyAI_Echo_SM_Clip(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="JoyAI_Echo_SM_Clip",
            display_name="JoyAI_Echo_SM_Clip",
            category="JoyAI_Echo_SM",
            inputs=[
                io.Combo.Input("clip",options= ["none"] + folder_paths.get_filename_list("clip") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf") ),
                io.Combo.Input("connector",options= ["none"] + folder_paths.get_filename_list("clip") ),
                io.Combo.Input("infer_device",options= ["cuda","cpu"] ),
            ],
            outputs=[io.Clip.Output(display_name="clip"),],
            )
    @classmethod
    def execute(cls,clip,gguf,connector,infer_device ) -> io.NodeOutput:
        clear_comfyui_cache()
        gemma_path=folder_paths.get_full_path("clip", clip) if clip != "none" else None
        gemma_gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        connector_path=folder_paths.get_full_path("clip", connector) if connector != "none" else None
        gemma_root=os.path.join(node_joyai_echo_path,"JoyAI_Echo/configs/gemma")
        clip=load_joyai_te(gemma_path or gemma_gguf_path,connector_path,gemma_root,torch.device(infer_device))
        return io.NodeOutput(clip)


class JoyAI_Echo_SM_Encoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoyAI_Echo_SM_Encoder",
            display_name="JoyAI_Echo_SM_Encoder",
            category="JoyAI_Echo_SM",
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input("prefetch_count",default=1,min=0,max=64),
                io.Boolean.Input("enable_streaming", default=False),
                io.String.Input("prompt",multiline=True,default=""),
                io.String.Input("prompt_files",multiline=False,default=""),
            ],
            outputs=[
                io.Conditioning.Output(display_name="te_cond"),
                ],
            )
    @classmethod
    def execute(cls,clip,prefetch_count,enable_streaming,prompt,prompt_files) -> io.NodeOutput:
        clear_comfyui_cache()
        prefetch_count=prefetch_count if prefetch_count > 0 else None
        clip.prefetch_count= prefetch_count
        clip.enable_streaming=enable_streaming
        if not prompt_files:
            if prompt:
                prompt_files=create_temp_json(prompt)
            else:
                raise Exception("No prompt or prompt_files")
        else:
            prompt_files=PureWindowsPath(prompt_files).as_posix()
            prompt_files=[prompt_files]      
        te_cond=infer_joyai_text(clip,prompt_files,device)
        torch.save(te_cond,os.path.join(folder_paths.get_output_directory(),"joy_echo_te_cond.pt"))
        return io.NodeOutput(te_cond)



