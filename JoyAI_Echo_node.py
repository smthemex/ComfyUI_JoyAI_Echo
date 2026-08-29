 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import time
import sys
import folder_paths
from comfy_api.latest import  io
import nodes
import math
from pathlib import PureWindowsPath
import argparse
from .JoyAI_Echo.inference import load_joyai_te,infer_joyai_text,load_joyai_engine,infer_joyai_video
from .JoyAI_Echo.inference_wm_causal import load_echo_wm_flash,infer_echo_wm_casusal
from .JoyAI_Echo.inference_wm import load_echo_wm,infer_echo_wm
from .node_utils import clear_comfyui_cache,create_temp_json,format_shot_num_secs,image2path
from .JoyAI_Echo.ltx_pipelines.causal_ti2vid import CausalTI2VidPipeline
from .JoyAI_Echo.ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
MAX_SEED = np.iinfo(np.int32).max

node_joyai_echo_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir
joyai_echo_node_instances={}

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
                io.Combo.Input("lora_1", options=["none"] + folder_paths.get_filename_list("loras") ),
                io.Float.Input("lora_1_weight", default=0, min=0, max=3, step=0.01),
                io.Combo.Input("lora_2", options=["none"] + folder_paths.get_filename_list("loras") ),
                io.Float.Input("lora_2_weight", default=0, min=0, max=3, step=0.01),
                io.Combo.Input("lora_3", options=["none"] + folder_paths.get_filename_list("loras") ),
                io.Float.Input("lora_3_weight", default=0, min=0, max=3, step=0.01),
                io.Combo.Input("lora_4", options=["none"] + folder_paths.get_filename_list("loras") ),
                io.Float.Input("lora_4_weight", default=0, min=0, max=3, step=0.01),
                io.Combo.Input("lora_5", options=["none"] + folder_paths.get_filename_list("loras") ),
                io.Float.Input("lora_5_weight", default=0, min=0, max=3, step=0.01),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls,dit,gguf,vae,audio_vae,lora_1, lora_1_weight,lora_2, lora_2_weight,lora_3, lora_3_weight,lora_4, lora_4_weight,lora_5, lora_5_weight) -> io.NodeOutput:
        clear_comfyui_cache()
        dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None 
        vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
        audio_vae_path=folder_paths.get_full_path("vae", audio_vae) if audio_vae != "none" else None
        echo_type="1.0"
        if dit_path :
            try:
                from safetensors import safe_open
                with safe_open(dit_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata()
                if metadata.get("architecture",None) :
                    echo_type="wm_base"
                elif metadata.get("merged_lora_rank",None) :
                    echo_type="wm_flash"

            except Exception as e:
                print(f"读取元数据失败: {e}")
        elif gguf_path:
            try:
                from gguf import GGUFReader
                reader = GGUFReader(gguf_path)
                fields = reader.fields
                for key, field in fields.items():
                    if key == 'architecture':
                        print(f"模型名称: {field.value}")
                        echo_type="wm_base"
                        break
                    elif key == 'merged_lora_rank':
                        print(f"模型类型: {field.value}")
                        echo_type="wm_flash" 
                        break
            except Exception as e:
                print(f"读取元数据失败: {e}")
        else:
            raise Exception("dit or gguf must be provided")
            
        custom_loras = []
        for lora_name, weight in [
            (lora_1, lora_1_weight), (lora_2, lora_2_weight), 
            (lora_3, lora_3_weight), (lora_4, lora_4_weight), 
            (lora_5, lora_5_weight)
        ]:
            if lora_name != "none" and weight > 0:
                lora_path = folder_paths.get_full_path("loras", lora_name)
                custom_loras.append({"path": lora_path, "weight": float(weight)})

        args = argparse.Namespace(
            config=(os.path.join(node_joyai_echo_path, "JoyAI_Echo/configs/inference.yaml") if echo_type == "1.0" 
                    else os.path.join(node_joyai_echo_path, "JoyAI_Echo/configs/inference_wm.yaml") if echo_type == "wm_base" 
                    else os.path.join(node_joyai_echo_path, "JoyAI_Echo/configs/inference_wm_causal.yaml")),
            device="cuda",
            dtype="bfloat16",
            checkpoint=dit_path or gguf_path,
            output_root=folder_paths.get_output_directory(),
            prompts_dir=folder_paths.get_output_directory(),
            prompts_glob="*.json",
            vae_path=vae_path,
            audio_vae_path=audio_vae_path,
            custom_loras=custom_loras,
            gemma_path="",
            video_local_attn_size=19,
            video_sink_size=7,
            video_chunk_size=3,

        )
        if echo_type == "1.0":
            model= load_joyai_engine(args)
        elif echo_type == "wm_base" :
            model= load_echo_wm(args,device)
        elif echo_type == "wm_flash" :
            model= load_echo_wm_flash(args,device)

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
                io.Int.Input("width", default=768, min=128, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=512, min=128, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Int.Input("num_frames", default=121, min=16, max=MAX_SEED,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("steps", default=30, min=16, max=MAX_SEED,step=1,display_mode=io.NumberDisplay.number),
                io.String.Input("shot_num_secs", default="", tooltip="example:  2.3, 5.8, 15"),
                io.Float.Input("frame_rate", default=25.0, min=8.0, max=120.0,step=1.0,display_mode=io.NumberDisplay.number),
                io.Int.Input("prefetch_count", default=1, min=0, max=48,step=1,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("enable_tiles", default=False),
                io.Int.Input("tile_size_in_frames", default=24, min=16, max=1024,step=8,display_mode=io.NumberDisplay.number),
                io.Int.Input("tile_size_in_pixels",default=512, min=64, max=4096,step=32,display_mode=io.NumberDisplay.number),
                io.Float.Input("fov_deg", default=70.0, min=10.0, max=120.0,step=1.0,display_mode=io.NumberDisplay.number),
                io.Combo.Input("streaming_mode",options= ["fast","swap","slow","auto"] ),
                io.Combo.Input("audio_memory_mode",options= ["center","max_response", "random"] ),
                io.Combo.Input("video_memory_mode",options= ["center","first", "random",] ),
                io.Combo.Input("geo_model",options= ["none"] + folder_paths.get_filename_list("geometry_estimation") ),
                io.String.Input("action_str", default="j-96,j-96,l-96,l-96",multiline=False, tooltip="example:  j-96,j-96,l-96,l-96"),
                io.Conditioning.Input("te_cond",optional=True),
                io.Conditioning.Input("negative_cond",optional=True),
                io.Video.Input("first_video",optional=True),
                io.Image.Input("image",optional=True),
            ], 
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
            ],
        )
    @classmethod
    def execute(cls, model,width,height,seed,num_frames,steps,shot_num_secs,frame_rate,prefetch_count,enable_tiles,tile_size_in_frames,tile_size_in_pixels,fov_deg,
                streaming_mode,audio_memory_mode, video_memory_mode,geo_model,action_str,te_cond=None,negative_cond=None,first_video=None,image=None) -> io.NodeOutput:
        clear_comfyui_cache()
        if te_cond is None:
            if not os.path.exists(os.path.join(folder_paths.get_output_directory(),"joy_echo_te_cond.pt")):
                raise Exception("te_cond is None or comfyUI outpu dont exist joy_echo_te_cond.pt  ")
            te_cond = torch.load(os.path.join(folder_paths.get_output_directory(),"joy_echo_te_cond.pt"),weights_only=False)

        model.streaming_mode=streaming_mode

        if isinstance(model, CausalTI2VidPipeline) or isinstance(model, TI2VidOneStagePipeline):
            geo_model_path=folder_paths.get_full_path("geometry_estimation", geo_model) if geo_model != "none" else None
            is_causal=True if  isinstance(model, CausalTI2VidPipeline) else False
            image_path=image2path(image,width, height) if image is not None else None
            args = argparse.Namespace(
                config=os.path.join(node_joyai_echo_path, "JoyAI_Echo/configs/inference_wm.yaml") if not is_causal else os.path.join(node_joyai_echo_path, "JoyAI_Echo/configs/inference_wm_causal.yaml"),
                num_frames=num_frames,
                width=width,
                height=height,
                fps=frame_rate,
                seed=seed,
                steps=steps,
                stg_scale=1.0,
                video_cfg=4.0,
                audio_cfg=2.0,
                stg_blocks=[29],
                guidance_scale=None,
                fov_deg=fov_deg,
                image=image_path,
                moge_model=geo_model_path,
                auto_fov=True if geo_model_path is not None else False,
                output=os.path.join(folder_paths.get_output_directory(), f"wm_{seed}_output{time.time():.0f}.mp4"),
                prompt="",
                negative_prompt="",
                no_audio=False ,
                action_overlay=False, # 不需要输出动作覆盖视频
                action_str=action_str,
                moge_python=sys.executable, # TODO: 添加 moge_python 参数 
                timesteps=None,
                translation_speed=None, # TODO: 添加 translation_speed 参数
                rotation_speed_deg=None, # TODO: 添加 rotation_speed_deg 参数
                pitch_limit_deg=None, # TODO: 添加 pitch_limit_deg 参数
                node_path=os.path.join(node_joyai_echo_path, "JoyAI_Echo"),
            )
            sample_rate=48000
            final_video=[]
            final_audio=[]
            from .JoyAI_Echo.helpers.action_condition import action_config
            model.action_config=action_config(width,height)
            neg_dict={}
            if negative_cond is None:
                if  os.path.exists(os.path.join(folder_paths.get_output_directory(),"joy_echo_te_cond_neg.pt")):
                    negative_cond = torch.load(os.path.join(folder_paths.get_output_directory(),"joy_echo_te_cond_neg.pt"),weights_only=False)
            if negative_cond is not None:
                for prompts_file,cached in negative_cond.items():
                    neg_dict = {
                        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in cached[0].items()
                    }
                    break
            for prompts_file,cached in te_cond.items():
                conditional_dict = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in cached[0].items()
                }
                if is_causal: 
                    video,audio=infer_echo_wm_casusal(args,model, conditional_dict,prefetch_count if prefetch_count > 0 else None,device)
                else:
                    conditional_dict["video_context_n"] = neg_dict.get("video_context", None)
                    conditional_dict["audio_context_n"] = neg_dict.get("audio_context", None)
                    video,audio=infer_echo_wm(args,model, conditional_dict,prefetch_count if prefetch_count > 0 else None,device)
                final_video.append(video)
                final_audio.append(audio.waveform)  #torch.Size([2, 50400])
               
                sample_rate=audio.sampling_rate
        
            print(f"[Inference] All {len(cached)} prompt file(s) processed.", flush=True)
            return  torch.cat(final_video,dim=0), {"waveform": torch.cat(final_audio,dim=-1).unsqueeze(0), "sample_rate": sample_rate}
        
        model.prefetch_count=prefetch_count if prefetch_count > 0 else None
        model.enable_tiles=enable_tiles
        model.tile_size_in_frames=tile_size_in_frames
        model.tile_size_in_pixels=tile_size_in_pixels

        cli_overrides = {"video_width": width,"video_height": height,"seed": seed,"num_frames": num_frames,"video_fps": frame_rate,} # "steps": steps,

        # 只有在成功解析出帧数列表时，才传递给 inference
        shot_frames_list=None or format_shot_num_secs(shot_num_secs) 
        if shot_frames_list is not None:
            cli_overrides["shot_num_frames"] = shot_frames_list

        images,audio=infer_joyai_video(model, te_cond,cli_overrides,first_video,audio_memory_mode, video_memory_mode)
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

        print(f"prompt_files is : {prompt_files}")
        if not prompt_files:
            if prompt:
                prompt_files=create_temp_json(prompt)
            else:
                raise Exception("No prompt or prompt_files")
        else:
            prompt_files=PureWindowsPath(prompt_files).as_posix()
            prompt_files=[prompt_files] 
        te_cond=prompt_files

        te_cond=infer_joyai_text(clip,prompt_files,device)
        torch.save(te_cond,os.path.join(folder_paths.get_output_directory(),"joy_echo_te_cond.pt"))
        return io.NodeOutput(te_cond)
    
from aiohttp import web
from server import PromptServer
import base64


@PromptServer.instance.routes.post("/joyai_echo/get_file_path")
async def get_file_path(request):
    try:
        data = await request.json()
        filename = data.get('filename', 'temp.json')
        content_base64 = data.get('content', '')
        if not content_base64:
            return web.json_response({"error": "No file content provided"}, status=400)
        file_content = base64.b64decode(content_base64)
        temp_dir = os.path.join(folder_paths.get_output_directory(), "joyai_temp")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        print(f"File saved to: {file_path}")

        return web.json_response({"path": file_path})
    except Exception as e:
        print(f"Error in get_file_path: {str(e)}")
        return web.json_response({"error": str(e)}, status=500)

