# ComfyUI_JoyAI_Echo
[JoyAI_Echo](https://github.com/jd-opensource/JoyAI-Echo) ：Pushing the Frontier of Long Video Generation  Standalone, inference-only release for minute-level multi-shot audio-video generation with a distilled DMD generator, paired cross-modal memory, and story-level consistency.

Update
----
* 2023.6.8 新增多个swap卸载模式,支持多个层的加卸载,修复tile无法使用的问题/add sawp unloading mode, support multiple layers of add and unload, fix the problem that tile cannot be used; 
* 2023.6.6 新增TE的safetesor 支持，目前只支持t2v，i2v需要官方的1.5版模型，在训练了，等等
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
  
text_encoder/dit :[JoyAI-Echo gguf or dit or clip ](https://huggingface.co/smthem/JoyAI-Echo-gguf)  
text_encoder/vae :[ltx2 text encoder vae audio vae...](https://huggingface.co/smthem/LTX-2.3-test-gguf)  

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
|     ├── gemma-3-12b-it-qat.safetensors #optional 可选
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