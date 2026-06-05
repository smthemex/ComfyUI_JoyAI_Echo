# ComfyUI_JoyAI_Echo
[JoyAI_Echo](https://github.com/jd-opensource/JoyAI-Echo) ：Pushing the Frontier of Long Video Generation  Standalone, inference-only release for minute-level multi-shot audio-video generation with a distilled DMD generator, paired cross-modal memory, and story-level consistency.

Update
----
* 复现官方代码， 3050 6G即可 跑5分钟 848*512 故事板长视频，无任何字幕，拼接自然
* just need 6G Vram to infer 5 minutes long video ，no word

1.Installation  
----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_JoyAI_Echo
```
2.requirements  
----
```
pip install -r requirements.txt
```
3.checkpoints 
----
  
[JoyAI-Echo gguf or dit ](https://huggingface.co/smthem/JoyAI-Echo-gguf)  
[ltx2 text encoder vae audio vae...](https://huggingface.co/smthem/LTX-2.3-test-gguf)  

```
├── ComfyUI/models/diffusion_models/
|     ├── JoyAI-Echo-transformer.safetensors #optional 可选
├── ComfyUI/models/vae/
|     ├── ltx-2.3-22b-distilled_video_vae.safetensors
|     ├── ltx-2.3-22b-distilled_audio_vae.safetensors
├── ComfyUI/models/gguf/
|     ├── gemma-3-12b-it-qat-Q4_0.gguf 
|     ├──JoyAI-Echo-Q8_0.gguf
|     ├──JoyAI-Echo-Q6_K.gguf
├── ComfyUI/models/clip/ 
|     ├── connector.safetensors  # or 11 version
```

4 Example
----

![](https://github.com/smthemex/ComfyUI_JoyAI_Echo/tree/main/example_workflows/example.png)

5 Citation
----

```
@techreport{echo2026longvideo,
  title        = {JoyAI-Echo: Pushing the Frontier of Long Video Generation},
  author       = {{Echo Team @ Joy Future Academy, JD}},
  institution  = {Joy Future Academy, JD},
  year         = {2026},
  month        = {May}
}
```