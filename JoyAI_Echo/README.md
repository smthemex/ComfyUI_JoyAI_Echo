<p align="center">
  <img src="assets/image.png" alt="JoyAI-Echo generated video gallery" width="100%">
</p>

<div align="center">

<h1>JoyAI-Echo</h1>

<p><strong>🎬 Pushing the Frontier of Long Video Generation</strong></p>

<p>Standalone, inference-only release for <strong>minute-level multi-shot audio-video generation</strong> with a distilled DMD generator, paired cross-modal memory, and story-level consistency.</p>

<p>
  <a href="https://www.researchgate.net/publication/405770309_JoyAI-Echo_Pushing_the_Frontier_of_Long_Audio-Visual_Generation"><b>📄 Paper</b></a> |
  <a href="https://echo-team-joy-future-academy-jd.github.io/Echo-LongVideo-Page/"><b>🌐 Project Page</b></a> |
  <a href="#quickstart"><b>🚀 Quickstart</b></a> |
  <a href="https://huggingface.co/jdopensource/JoyAI-Echo"><b>🤗 Hugging Face</b></a> |
  <a href="#results"><b>📊 Results</b></a> |
  <a href="#citation"><b>📝 Citation</b></a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.8-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.8">
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 12.8">
  <img src="https://img.shields.io/badge/Release-Inference--Only-black?style=flat-square" alt="Inference">
  <img src="https://img.shields.io/badge/Long%20Video-5%20min-d61f2c?style=flat-square" alt="5 minute long video">
</p>

</div>

## Abstract

Long video generation still suffers from error accumulation, weak temporal coherence, and prohibitive latency, limiting its applicability to interactive scenarios. We present **JoyAI-Echo**, a framework that breaks these barriers through four key advances.
Central to its performance, a cross-modal audio-visual memory bank preserves character appearance and voice timbre consistently over five-minute videos, while a post-training pipeline combines memory-based reinforcement learning with distribution matching distillation for a **7.5× speedup** to substantially boost visual quality and alignment.
Empowered by these two components, **JoyAI-Echo** decisively outperforms *HappyOyster* (directing mode) on long-form generation and even surpasses the short-video specialist *Wan 2.6* on human-centric tasks.
Beyond raw generation quality, an interactive agent enables real-time user editing through conversational instructions, and a lightweight super-resolution module maintains high definition under streaming latency, further elevating the overall experience and delivering instantly editable, conversation-speed video creation.
For the first time, **JoyAI-Echo** simultaneously achieves long-range cross-modal consistency, real-time inference for minute-long video, conversational interactivity, and high-resolution output — without compromise, inaugurating a new era of interactive video generation.
Codes and weights will be open-sourced.

## Highlights

- 🎞️ **Minute-level multi-shot stories**: generate a sequence of coherent shots from one prompt JSON.
- ⚡ **DMD-distilled few-step inference**: ~7.5x faster than the original pipeline.
- 🔊 **Joint audio-video generation**: one pipeline produces synchronized video and audio.
- 🧠 **Paired cross-modal memory bank**: conditions each new shot on prior visual identity and voice context for story-level consistency.

## Demo Gallery

Explore long-form and short-form JoyAI-Echo cases on the [Project Page](https://echo-team-joy-future-academy-jd.github.io/Echo-LongVideo-Page/). 🍿

## Results

### Reported Scale

| Item | Value |
| --- | ---: |
| 🎬 Long-form coherent story length | **5 min** |
| ⚡ Generation speedup over the original multi-step pipeline | **7.5x** |
| 📚 Benchmark stories | **100** |
| 🎞️ Generated evaluation shots | **3,000** |
| 🕒 Frames per shot | **241 @ 25 fps** |

### Human Evaluation

GSB user study on long- and short-video generation. The numbers denote the percentage of user preferences.

| Aspect<br>(Long Video) | JoyAI-Echo | Tie | HappyOyster<br> (Directing) | 
| --- | ---: | ---: | ---: | 
| Visual aesthetics | **63.6%** | 8.8% | 27.6% | 
| Audio quality | **81.7%** | 6.5% | 11.8% |
| Prompt following | **80.6%** | 13.5% | 5.9% | 
| IP consistency | **59.4%** | 12.9% | 27.7% |

| Aspect<br>(Short Video) | JoyAI-Echo | Tie | Wan 2.6 |
| ---  | ---: | ---: | ---: |
| Visual aesthetics | **58.8%** | 14.7% | 26.5% |
| Audio quality  | 32.3% | 30.9% | 36.8% |
| Prompt following | 33.8% | 36.8% | 29.4% |

## Repository Layout

```text
.
+-- configs/
|   `-- inference.yaml                # all inference parameters (YAML)
+-- checkpoints/                      # model weights (download separately)
|   +-- echo-longvideo-release.safetensors
|   `-- gemma-3-12b/
+-- prompts/                          # multi-shot prompt JSON files
|   +-- example_single_shot.json
|   `-- example_multi_shot.json
+-- ltx-core/src/ltx_core/            # transformer, VAE, text-encoder building blocks
+-- ltx-pipelines/src/ltx_pipelines/  # sampler and pipeline utilities
+-- ltx-distillation/
|   +-- src/ltx_distillation/         # DMD wrappers, AV pipelines, memory bank, utils
|   `-- scripts/multishot_inference_dmd.py
+-- inference.py                      # main entrypoint (load once, infer all)
+-- requirements.txt
`-- environment.yml
```

## Quickstart

### 1. Clone

```bash

git clone https://github.com/jd-opensource/JoyAI-Echo.git
cd JoyAI-Echo
```

### 2. Create the environment

The reference environment is **Python 3.11 + PyTorch 2.8 + CUDA 12.8**.

With conda:

```bash
conda env create -f environment.yml
conda activate echo-long
```

With `uv`:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
```

[`ffmpeg`](https://ffmpeg.org/download.html) must be available on `PATH` for shot concatenation. The conda recipe includes it. If you use `uv`, install it with your system package manager:

```bash
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
```

### 3. Download checkpoint

Download the JoyAI-Echo release checkpoint and Gemma text encoder:

| File | Description | Size | Link |
| --- | --- | --- | --- |
| `echo-longvideo-release.safetensors` | Full model (transformer + VAE + vocoder) | ~46 GB |[`JoyAI-Echo`](https://huggingface.co/jdopensource/JoyAI-Echo)  |
| `gemma-3-12b/` | Instruction-tuned model (text encoder) | ~24 GB | [`gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) |

Place them under `checkpoints/`:

```text
checkpoints/
+-- echo-longvideo-release.safetensors
`-- gemma-3-12b/
```

### 4. Write a story prompt

Create a JSON file under `prompts/`. Each file is a single object with a `prompts` list, where **every string is one complete shot**. A single string produces one shot; multiple strings produce a multi-shot story, with each new shot conditioned on the previous ones through the paired audio-video memory bank.

Inside each string, write these parts in order:

| Part | What to describe |
| --- | --- |
| **Roles & Subjects** | Describe the appearance of all visible people, including age, build, hair, face, wardrobe, and speaking voice timbre when applicable. |
| **Action & Dialogue** | What the subject does and speaks. |
| **Style** | The overall visual and emotional aesthetic — e.g. realistic motorsport film language, cool daylight, restrained cinematic tension. |
| **Camera Movement** | The shot type and framing or movement — e.g. a stable close-up on the face, or a medium shot from the waist up. |
| **Background** | The setting and scene details behind the subject. |
| **Sound Effects & BGM** | The sounds in the scene and the background music — e.g. room tone, wind, footsteps and fabric, with a soft low music bed under the dialogue or nobackground music |

To turn a story into these shot descriptions automatically, pair an LLM with the story-writer system prompt at `prompts/story_writer_system_prompt.md`. A more convenient prompt-writing workflow will be released as a **director agent** for everyone to use.

### 5. Run inference

```bash
python inference.py
```

This loads the model once and processes all prompt files under `prompts/`.

> 💡 **Note**: The inference pipeline is optimized to run on lower-VRAM
> GPUs. Peak GPU usage is around **46–50 GB**, at the cost of slightly
> longer per-shot inference time.

Outputs are written to:

```text
inference_result/outputs/<prompt-name>/inference_<timestamp>/
```

## Configuration

All inference parameters are managed in `configs/inference.yaml`. The file is organized into sections:

| Section | Contents |
| --- | --- |
| `paths` | Checkpoint path, prompts directory, output root |
| `video` | Resolution, frame count, FPS, seed |
| `denoising` | Step list and sigma schedule |
| `memory` | Memory bank size, save mode, LoRA settings |
| `audio_memory` | Audio window, mel-spectrogram params |
| `inference` | Device, dtype, grad scale |

### Override via CLI

Any YAML parameter can be overridden from the command line:

```bash
python inference.py --seed 42 --num-frames 121 --video-height 480 --video-width 832
```

Use a custom config file:

```bash
python inference.py --config configs/my_experiment.yaml
```

The Python entrypoint exposes the full configuration surface:

```bash
python inference.py --help
```

## Hardware

Peak GPU usage is around **46–50 GB** for the default **25 fps x 241 frames x 1280 x 736** setting, so a single H100/A100-class (80 GB) or 48 GB GPU is sufficient.

For smaller GPUs, reduce resolution/frames:

```bash
python inference.py --num-frames 121 --video-height 480 --video-width 832
```

## TODO List

- [x] Release inference code
- [x] Release model checkpoints
- [x] Add prompt examples
- [ ] Release Echo-SR (Super-resolution)
- [ ] Release Director Agent 

## Links

- Project page: [`https://echo-team-joy-future-academy-jd.github.io/Echo-LongVideo-Page/`](https://echo-team-joy-future-academy-jd.github.io/Echo-LongVideo-Page/)
- Repository: [`https://github.com/jd-opensource/JoyAI-Echo`](https://github.com/jd-opensource/JoyAI-Echo)
- huggingface: [`https://huggingface.co/jdopensource/JoyAI-Echo`](https://huggingface.co/jdopensource/JoyAI-Echo)

## Acknowledgements

We gratefully acknowledge the open-source projects this work builds upon — in particular [LTX2.3](https://huggingface.co/Lightricks/LTX-2.3) for the base video generator and [Gemma](https://huggingface.co/google/gemma-3-12b-it) for the text encoder. Thanks to the broader research community whose contributions made this release possible.

**For academic research and non-commercial use only.**

## Citation

If JoyAI-Echo helps your research or products, please cite:

```bibtex
@techreport{echo2026longvideo,
  title        = {JoyAI-Echo: Pushing the Frontier of Long Video Generation},
  author       = {{Echo Team @ Joy Future Academy, JD}},
  institution  = {Joy Future Academy, JD},
  year         = {2026},
  month        = {May}
}
```

## License

This project is based on LTX-2 by Lightricks Ltd.

Portions of the original LTX-2 codebase have been modified by JD.com for academic and research purposes only. 
This project is not intended for commercial use. For commercial use of LTX-2 or its derivatives, please contact Lightricks Ltd.

All original copyright, license, patent, trademark, and attribution notices from LTX-2 are retained. 
This project remains subject to the LTX-2 Community License Agreement.
