import librosa
import numpy as np
import pyloudnorm as pyln
from univoc import Vocoder
import torch
import soundfile as sf
from librosa.filters import mel
from scipy import signal
from scipy.signal import get_window
from numpy.random import RandomState
from collections import OrderedDict
from math import ceil

from model_bl import D_VECTOR
from model_vc import Generator

# Melc vocoder
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

meter = pyln.Meter(16000)

def wav2melV2(wav):
    loudness = meter.integrated_loudness(wav)
    wav = pyln.normalize.loudness(wav, loudness, -24)
    peak = np.abs(wav).max()
    if peak >= 1:
        wav = wav / peak * 0.999
    mel = melspectrogram(wav, n_mels=80)
    mel = np.transpose(mel, (1, 0))
    return mel

# Mel speacker encoder

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)  



mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

def prepare_spectrogram(path, rd_int=None):
    x, fs = sf.read(path)
    y = signal.filtfilt(b, a, x)
    if rd_int is None:
        rd_int = int(path.split('/')[-2][1:])
    prng = RandomState(rd_int) # cosa vuol dire?
    wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)  
    S = S.astype(np.float32)
    return S

# Speacker encoder
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load('3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
num_uttrs = 10
len_crop = 128


def process_speacker(tmp):
    left = np.random.randint(0, tmp.shape[0]-len_crop)
    melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
    emb = C(melsp)
    return emb.detach().squeeze().cpu().numpy()


# AUTOVC
def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

def prepare_input(s1, emb1, emb2, G):
    x_org, len_pad = pad_seq(s1)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(emb1[np.newaxis, :]).to(device)
    
    emb_trg = torch.from_numpy(emb2[np.newaxis, :]).to(device)
    
    with torch.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
    if len_pad == 0:
        uttr_trg = x_identic_psnt[0, 0, :, :]
    else:
        uttr_trg = x_identic_psnt[0, 0, :-len_pad, :]
    return uttr_trg



# DATA
device = 'cuda:0'
path1="wavs/i300/galatea_01_barrili_f000001.wav"
path2="wavs/i301/imalavoglia_00_verga_f000002.wav"

# PreProcessing
wav1, _ = librosa.load(path1, sr=16000)
vocoder_mel1 = wav2melV2(wav1)
speacker_encoder_mel1 = prepare_spectrogram(path1, rd_int=0)

wav2, _ = librosa.load(path2, sr=16000)
vocoder_mel2 = wav2melV2(wav2)
speacker_encoder_mel2 = prepare_spectrogram(path2, rd_int=1)

# Encod
emb1 = process_speacker(speacker_encoder_mel1)
emb2 = process_speacker(speacker_encoder_mel2)

# AutoVC
G = Generator(32,256,512,32).eval().to(device)
model_path = '/home/super/Models/autovc_simple/generator_run2.pth'
g_checkpoint = torch.load(model_path, map_location=device)
G.load_state_dict(g_checkpoint)

spect_vc1 = prepare_input(vocoder_mel1, emb1, emb2, G)

# vocoder
vocoder = Vocoder.from_pretrained(
    "https://github.com/bshall/UniversalVocoding/releases/download/v0.2/univoc-ljspeech-7mtpaq.pt",
).cuda()

torch_mel = spect_vc1.unsqueeze(dim=0)
torch_mel = torch_mel.cuda()
with torch.no_grad():
    wav2, sr = vocoder.generate(torch_mel)

path2= "out.wav"
sf.write(path2, wav2, sr)

