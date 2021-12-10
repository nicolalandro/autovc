import os

import librosa
import numpy as np
import pyloudnorm as pyln
from numpy.random import RandomState


def melspectrogram(
        wav,
        sr=16000,
        hop_length=200,
        win_length=800,
        n_fft=2048,
        n_mels=128,
        fmin=50,
        preemph=0.97,
        top_db=80,
        ref_db=20,
):
    mel = librosa.feature.melspectrogram(
        librosa.effects.preemphasis(wav, coef=preemph),
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        norm=1,
        power=1,
    )
    logmel = librosa.amplitude_to_db(mel, top_db=None) - ref_db
    logmel = np.maximum(logmel, -top_db)
    return logmel / top_db


# audio file directory
rootDir = './wavs'
# spectrogram directory
targetDir = './spmel_other'

dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)
meter = pyln.Meter(16000)

for subdir in sorted(subdirList):
    print(subdir)
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    _, _, fileList = next(os.walk(os.path.join(dirName, subdir)))
    prng = RandomState(int(subdir[1:]))
    for fileName in sorted(fileList):
        path1 = os.path.join(dirName, subdir, fileName)
        # Read audio file
        wav, _ = librosa.load(path1, sr=16000)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -24)
        peak = np.abs(wav).max()
        if peak >= 1:
            wav = wav / peak * 0.999
        mel = melspectrogram(wav, n_mels=80)
        mel = np.transpose(mel, (1, 0))

        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                mel.astype(np.float32), allow_pickle=False)
