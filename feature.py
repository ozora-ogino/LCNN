import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import scipy



def preEmphasis(wave, p=0.97):
    return scipy.signal.lfilter([1.0, -p], 1, wave)


def get_fft(df, path):
    """
    This function extracts spectrograms from raw audio data by using FFT.

    Args:
     df:
      This augument should be Pandas DataFrame and extracted from ASVspoof2019 protocol.
     path:
      Path to database of ASVSpoof2019
    
    Returns: 
     data:
      spectrograms that have 4 dimentions
     label:
      0 = Genuine, 1 = Spoof
    """

    data = []
    for audio in tqdm(df['utt_id']):
        file = path + audio + '.flac'
        #load audio file
        wave, sr = librosa.load(file)
        wave = preEmphasis(wave)
        steps = int(len(wave) * 0.0081)
        #calculate STFT
        S_F = librosa.stft(wave,
                            n_fft=sr,
                            win_length=1700,
                            hop_length=steps,
                            window='blackman')
        amp_db = librosa.amplitude_to_db(np.abs(S_F), ref=np.max)
        amp_db = amp_db[:800,:].astype('float32')
        data.append(amp_db)
    data = np.array(data)
    print(data.shape)

    #reshape for using with CNN .
    data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

    label = np.ones(len(df))
    label[df['key'] == 'bonafide'] = 0
    
    return data, label.astype(int)


def get_cqt(df, path):
    """

    This function extracts spectrograms from raw audio data by using CQT.

    
    Plsease refer to get_fft's auguments and outputs.
    They are almost same.
    """
    samples = df['utt_id']
    max_len = 200 # for resizing cqt spectrogram.

    for i, sample in enumerate(tqdm(samples)):
        full_path = path + sample + '.flac'
        y, sr = librosa.load(full_path)
        y = preEmphasis(y)
        cq = librosa.core.cqt(y, sr=sr)
        cq_db = librosa.amplitude_to_db(np.abs(cq)) # Amplitude to dB.
        shape_1 = cq_db.shape[0]

        if i == 0:
            resized_data = np.zeros((len(df), shape_1, max_len))

        if max_len <= cq_db.shape[1]:
            cq_db = cq_db[:,:max_len]

        else:
            diff = max_len - cq_db.shape[1]
            zeros = np.zeros((shape_1, diff))
            cq_db = np.concatenate([cq_db, zeros], 1)

        resized_data[i] = np.float32(cq_db)

    label = np.ones(len(df))
    label[df['key'] == 'bonafide'] = 0

    return resized_data, label.astype(int)