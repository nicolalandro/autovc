# AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss
This repo is a fork of [autovc](https://github.com/auspicious3000/autovc) and it aim to write a usable demo of this models.

## Run Notebooks
- Install dependencies
- download models
- run jupyter
- Demo.ipynb: take two audio and generate the new one with the voice of the second and the words of the first

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

