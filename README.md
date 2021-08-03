[![License: MIT](https://img.shields.io/badge/license-MIT-lightgray)](LICENSE) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolalandro/autovc/blob/master/AutoVCDemoColab.ipynb)

# AUTOVC
This repo is a fork of [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://github.com/auspicious3000/autovc). 
It aim to write an easy usable demo of this models.

## Run Notebooks
* Use local
  * Install dependencies
  * clone the project
  * download models into cloned folder
  * run jupyter
  * Demo.ipynb: take two audio and generate the new one with the voice of the second and the words of the first

## Dependencies
- Python 3
- jupyter
- Numpy
- PyTorch >= v0.4.1
- wavenet_vocoder ```pip install wavenet_vocoder```
  for more information, please refer to https://github.com/r9y9/wavenet_vocoder
- librosa
- soundfile
- scipy
- tqdm
- (matplotlib ?)

## Pre-trained models
* [AUTOVC](https://github.com/nicolalandro/autovc/releases/download/0.1/autovc.ckpt)
* [Speacker Encoder](https://github.com/nicolalandro/autovc/releases/download/0.1/3000000-BL.ckpt)
* [WaveNet Vocoder](https://github.com/nicolalandro/autovc/releases/download/0.1/checkpoint_step001000000_ema.pth)

