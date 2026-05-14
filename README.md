# Improving Calibration in Test-Time Prompt Tuning for Vision-Language Models via Data-Free Flatness-Aware Prompt Pretraining [CVPR 2026]

[![Conference](https://img.shields.io/badge/CVPR-2026-0b5fff.svg)](https://cvpr.thecvf.com/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-4b9e5d.svg)](https://arxiv.org/abs/2604.27715)

This repository contains the official implementation of our CVPR 2026 paper:
> [**Improving Calibration in Test-Time Prompt Tuning for Vision-Language Models via Data-Free Flatness-Aware Prompt Pretraining**](https://arxiv.org/abs/2604.27715)  
> **Hyeonseo Jang**, **Jaebyeong Jeon**, **Joong-Won Hwang**, and **Kibok Lee**

## 📖 Overview

**FPP (Flatness-aware Prompt Pretraining)** is a simple yet effective pretraining framework for test-time prompt tuning (TPT) of vision-language models that initializes prompts within flatter regions of the loss landscape prior to adaptation.

- 🔹 A **flatness loss** guides the prompt toward flat minima, which are associated with improved calibration.
- 🔹 An **alignment loss** keeps the learned text features close to those of the original prompt, preserving its semantic structure.
- 🔹 FPP can be seamlessly integrated with existing TPT-based methods by replacing their initial prompts, improving both calibration and accuracy while requiring no external resources.

## ⚙️ Installation

```bash
conda env create -f environment.yml
conda activate fpp
```

## 📂 Datasets

FPP is evaluated on 10 fine-grained classification datasets (StanfordCars, OxfordPets, Flowers102, Food101, DTD, SUN397, Caltech101, UCF101, EuroSAT, FGVCAircraft) and 4 ImageNet variants widely used to assess robustness under natural distribution shifts (ImageNet-A, ImageNet-V2, ImageNet-R, ImageNet-Sketch). We follow the dataset structure of [DIKI](https://github.com/lloongx/DIKI); please refer to their [dataset instructions](https://github.com/lloongx/DIKI/blob/main/docs/datasets.md) for setup details.

Before running any script, set `data_root` at the top of the shell scripts (or export `DATA_ROOT`) to the path of your dataset root directory.

## 🚀 Training

### Fine-Grained Classification (FG)
Run FPP on 10 fine-grained datasets with the hard textual template `"a photo of a"` as the predefined prompt:
```bash
bash FPP_FG.sh
```

### Distribution Shift (DS)
Run FPP on 4 ImageNet variants under natural distribution shifts:
```bash
bash FPP_DS.sh
```

### CoOp Initialization
Run FPP on 10 fine-grained datasets with the pretrained prompt instead of the hard textual template:
```bash
bash FPP_FG_COOP.sh
```

### Baselines
Run the baseline calibration methods — [C-TPT](https://arxiv.org/abs/2403.14119) and [O-TPT](https://arxiv.org/abs/2503.12096):
```bash
bash CTPT_FG.sh   # C-TPT on fine-grained datasets
bash CTPT_DS.sh   # C-TPT on distribution-shift datasets
bash OTPT_FG.sh   # O-TPT on fine-grained datasets
bash OTPT_DS.sh   # O-TPT on distribution-shift datasets
```

## 📝 Citation
If you find this work useful, please consider citing our paper:
```bibtex
@inproceedings{jang2026fpp,
  title={Improving Calibration in Test-Time Prompt Tuning for Vision-Language Models via Data-Free Flatness-Aware Prompt Pretraining},
  author={Jang, Hyeonseo and Jeon, Jaebyeong and Hwang, Joong-Won and Lee, Kibok},
  booktitle={CVPR},
  year={2026}
}
```

## 📬 Contact

For questions or issues, please contact [jhyeonseo715@yonsei.ac.kr](mailto:jhyeonseo715@yonsei.ac.kr).

## 🙏 Acknowledgements

This codebase builds on [O-TPT](https://github.com/ashshaksharifdeen/O-TPT). We thank the authors for releasing their code.
