<div align="center">

# ReImagine: Rethinking Controllable High-Quality Human Video Generation via Image-First Synthesis

<p>
  <a href="https://arxiv.org/abs/2604.19720">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2604.19720-15803d?logo=arxiv&logoColor=white&style=flat" alt="Paper on arXiv" />
  </a>
  <a href="https://keruzheng.github.io/ReImagine-Project/">
    <img src="https://img.shields.io/badge/Project-Website-4f46e5?logo=googlechrome&logoColor=white&style=flat" alt="Project page" />
  </a>
  <a href="https://taited-reimagine.hf.space/">
    <img src="https://img.shields.io/badge/Demo-Hugging%20Face-ffb900?logo=huggingface&logoColor=78350f&style=flat" alt="Hugging Face Space" />
  </a>
  <a href="https://www.youtube.com/watch?v=M5J1Yfp778o">
    <img src="https://img.shields.io/badge/Video-YouTube-ff0000?logo=youtube&logoColor=white&style=flat" alt="Demo on YouTube" />
  </a>
</p>


<a href="https://keruzheng.github.io/ReImagine-Project/">
  <img src="assets/teaser.png" alt="ReImagine — controllable human video generation via image-first synthesis" width="60%" />
</a>

**[Try it online Here: https://taited-reimagine.hf.space/](https://taited-reimagine.hf.space/)**

</div>

## Overview

This repository hosts the official implementation of **ReImagine**, a framework for controllable high-quality human video generation via **image-first synthesis**. For more context, see the [paper on arXiv](https://arxiv.org/abs/2604.19720) and the [project website](https://keruzheng.github.io/ReImagine-Project/).

## What's New 🚀

- **April 23 2026**: Updated the Image-First Synthesis demo.
- **April 22 2026**: Initial repository launch.

Stay tuned for further updates!

## Getting Started

### Environment

We develop and test with **Python 3.10**, **PyTorch 2.4.1**, and **CUDA 12.4**. Install the CUDA 12.4 PyTorch wheels, then install this package in editable mode:

```bash
conda create -n reimagine python=3.10
conda activate reimagine
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

### Pretrained Base Models

Download checkpoints with the Hugging Face Hub CLI ([`hf download`](https://huggingface.co/docs/huggingface_hub/guides/cli) or `huggingface-cli download` on older installs). For **FLUX.1-Kontext-dev**, you can **skip the monolithic** `flux1-kontext-dev.safetensors` and the **`vae/`** tree:

```bash
hf download black-forest-labs/FLUX.1-Kontext-dev \
  --local-dir ./models/FLUX.1-Kontext-dev \
  --exclude "flux1-kontext-dev.safetensors" \
  --exclude "vae/**"
```

For ControlNet:

```bash
hf download jasperai/Flux.1-dev-Controlnet-Surface-Normals \
  --local-dir ./models/Flux.1-dev-Controlnet-Surface-Normals
```

### ReImagine LoRA Weights

ReImagine LoRA weight files are hosted on Hugging Face at **[taited/ReImagine-Pretrained](https://huggingface.co/taited/ReImagine-Pretrained/)**.

| **SMPL-X Params** | **Input Type**                             | **File**                                                                                                                              | **Status** |
| ----------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| w/o               | Canonical human (front & back views)       | [`kontext-wo_smplx-lora.safetensors`](https://huggingface.co/taited/ReImagine-Pretrained/blob/main/kontext-wo_smplx-lora.safetensors) | Available  |
| w/o               | Disentangled assets (face, clothes, shoes) | *TBA*                                                                                                                                 | Planned    |

**Download:** Use the same [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) as for the base models:

```bash
hf download taited/ReImagine-Pretrained --local-dir ./models/ReImagine-Pretrained
```

### Inference with Image-First Synthesis (`inference_img.py`)

Once you have prepared the pretrained weights, use [`inference_img.py`](inference_img.py) to infer each frame. This script requires two image inputs: a **wide reference image** (left = front, right = back) and a **normal map**. The normal map is generated from SMPL-X's global coordinate system based on camera parameters.

For more details on the usage of `inference_img.py`, check the full guide and example.

### Inference with Temporal-Refinement Video Synthesis (To be continued)

The code for **Temporal-Refinement Video Synthesis** is currently being organized for open-source release. Once available, it will allow inference on video data with temporal refinement.

Stay tuned for updates!

## Status

### Released
* [x] Code for **Image-First Synthesis** inference (`inference_img.py`)
* [x] Pretrained **LoRA** weights (available for download)
* [x] Documentation and usage instructions for basic inference

### To be released
* [ ] Code for **Temporal-Refinement Video Synthesis**
* [ ] Pretrained model weights for **Disentangled assets** (face, clothes, shoes)
* [ ] Full dataset release

We are actively organizing and updating the repository. Updates will be added here as each item becomes available.

## Acknowledgments

This repository’s implementation is **based on** [DiffSynth Studio](https://github.com/modelscope/diffsynth-studio) (ModelScope). We thank the authors and maintainers for releasing their work. The upstream project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

We acknowledge the contributions of the teams behind **[FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)** and **[Flux.1-dev-Controlnet-Surface-Normals](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Surface-Normals)** for their open-source releases that this project builds on.

## Citation

If you find this project useful, please consider citing our paper:

```bibtex
@article{sun2025rethinking,
  title={ReImagine: Rethinking Controllable High-Quality Human Video Generation via Image-First Synthesis},
  author={Sun, Zhengwentai and Zheng, Keru and Li, Chenghong and Liao, Hongjie and Yang, Xihe and Li, Heyuan and Zhi, Yihao and Ning, Shuliang and Cui, Shuguang and Han, Xiaoguang},
  journal={arXiv preprint arXiv:2604.19720},
  year={2026},
  url={https://arxiv.org/abs/2604.19720v1}
}
```
