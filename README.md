[![License: MIT](https://img.shields.io/badge/license-MIT-lightgray)](LICENSE) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolalandro/autovc/blob/master/AutoVCDemoColab.ipynb)

# AUTOVC
This repo is a fork of [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://github.com/auspicious3000/autovc). 
It aim to write an easy usable demo of this models.

## Run Notebooks
* Open on Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolalandro/autovc/blob/master/AutoVCDemoColab.ipynb)
* Use in local environment (with GPU: cuda):
  * Install dependencies
  * clone the project
  * download models into cloned folder
  * run jupyter
  * Demo.ipynb: take two audio and generate the new one with the voice of the second and the words of the first
* Train on Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolalandro/autovc/blob/master/TrainAutoVC.ipynb)

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

## Train on VoxCeleb
```
# make spect files (do it one times)
nohup python make_spect_for_vox_cel.py \
    --root-dir="/home/super/datasets-nas/Vox2celeb/vox2celeb-1/wav" \
    --target-dir="/home/super/datasets-nas/Vox2celeb/vox2celeb-1/spmel" \
    > make_spec.log 2>&1 &!

# make train.pkl file (do it one times)
wget https://github.com/nicolalandro/autovc/releases/download/0.1/3000000-BL.ckpt
CUDA_VISIBLE_DEVICES="0"  nohup python make_metadata.py --root-dir="/home/super/datasets-nas/Vox2celeb/vox2celeb-1/spmel" \
    > make_metadata.log 2>&1 &!

!wget https://github.com/nicolalandro/autovc/releases/download/0.1/autovc.ckpt
CUDA_VISIBLE_DEVICES="0"  nohup python main.py --data_dir="/home/super/datasets-nas/Vox2celeb/vox2celeb-1/spmel" \
    --outfile-path="/home/super/Models/autovc_voxceleb/generator.pth" \
    --num_iters 10000 --batch_size=10 --dim_neck 32 --dim_emb 256 --dim_pre 512 --freq 32 --pretrained "autovc.ckpt" \
     > train.log 2>&1 &!
```

## Train with new vocoder
```
python3.8 make_spect.py # create folder spmel
python3.8 make_spect_other_vocoder.py # create the folder spmel_other
CUDA_VISIBLE_DEVICES="0" python3.8 make_metadata.py --root-dir="./spmel" # create the spmel/train.pkl # use speaker encoder on /spmel
cp spmel/train.pkl spmel_other # copy the spmel/train.pkl into spmel_other/train.pkl
CUDA_VISIBLE_DEVICES="0" python3.8 main.py --data_dir="spmel_other" \
    --outfile-path="/home/super/Models/autovc_simple/generator.pth" \
    --num_iters 10000 --batch_size=6 --dim_neck 32 --dim_emb 256 --dim_pre 512 --freq 32
CUDA_VISIBLE_DEVICES="0" python3.8 test_audio.py
```

