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

| AUTOVC | Speaker Encoder | WaveNet Vocoder |
|----------------|----------------|----------------|
| [link](https://drive.google.com/file/d/1SZPPnWAgpGrh0gQ7bXQJXXjOntbh4hmz/view?usp=sharing)| [link](https://drive.google.com/file/d/1ORAeb4DlS_65WDkQN6LHx5dPyCM5PAVV/view?usp=sharing) | [link](https://drive.google.com/file/d/1Zksy0ndlDezo9wclQNZYkGi_6i7zi4nQ/view?usp=sharing) |


