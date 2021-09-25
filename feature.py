from typing import Tuple, Callable
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import scipy
import pickle


def save_feature(feature, path):
    with open(path, "wb") as web:
        pickle.dump(feature, web, protocol=4)


def preEmphasis(wave: np.ndarray, p=0.97) -> np.ndarray:
    """Pre-Emphasis"""
    return scipy.signal.lfilter([1.0, -p], 1, wave)


def get_stft(protocol_df: pd.DataFrame, path: str) -> Tuple[np.ndarray, np.ndarray]:
    """

    This function extracts spectrograms from raw audio data by using FFT.

    Args:
     protocol_df(pd.DataFrame): ASVspoof2019 protocol.
     path(str): Path to ASVSpoof2019

    Returns:
     data: spectrograms that have 4 dimentions
     label: 0 = Genuine, 1 = Spoof


    """

    data = []
    for audio in tqdm(protocol_df["utt_id"]):
        file = path + audio + ".flac"
        # load audio file
        wave, sr = librosa.load(file)
        wave = preEmphasis(wave)
        steps = int(len(wave) * 0.0081)
        # calculate STFT
        S_F = librosa.stft(
            wave, n_fft=sr, win_length=1700, hop_length=steps, window="blackman"
        )
        amp_db = librosa.amplitude_to_db(np.abs(S_F), ref=np.max)
        amp_db = amp_db[:800, :].astype("float32")
        data.append(amp_db)
    data = np.array(data)[..., np.newaxis]
    print(data.shape)

    label = np.ones(len(protocol_df))
    label[protocol_df["key"] == "bonafide"] = 0

    return data, label.astype(int)


def get_cqt(protocol_df: pd.DataFrame, path: str) -> Tuple(np.ndarray, np.ndarray):
    """

    This function extracts spectrograms from raw audio data by using CQT.


    Plsease refer to get_fft's auguments and outputs.
    They are almost same.

    """

    samples = protocol_df["utt_id"]
    max_len = 200  # for resizing cqt spectrogram.

    for i, sample in enumerate(tqdm(samples)):
        full_path = path + sample + ".flac"
        y, sr = librosa.load(full_path)
        y = preEmphasis(y)
        cq = librosa.core.cqt(y, sr=sr)
        cq_db = librosa.amplitude_to_db(np.abs(cq))  # Amplitude to dB.
        shape_1 = cq_db.shape[0]

        if i == 0:
            resized_data = np.zeros((len(protocol_df), shape_1, max_len))

        if max_len <= cq_db.shape[1]:
            cq_db = cq_db[:, :max_len]

        else:
            diff = max_len - cq_db.shape[1]
            zeros = np.zeros((shape_1, diff))
            cq_db = np.concatenate([cq_db, zeros], 1)

        resized_data[i] = np.float32(cq_db[..., np.newaxis])

    label = np.ones(len(protocol_df))
    label[protocol_df["key"] == "bonafide"] = 0

    return resized_data, label.astype(int)


def get_fft_delta(
    path_to_data: str, extractor: Callable, df: pd.DataFrame, saving_path: str
):
    """get_fft_delta

    Extract and save fft, delta and delta2 features

    Args:
        path_to_data(str): The full path to the directory that holds ASV2019's audio data.
        extractor(function): The function that returns spectrograms.
                             Its format should be same as get_fft or get_cqt.
        df(pd.DataFrame): ASVspoof2019's data protocol. This helps to define the name of audio files.
        saving_path(str): An full path for saving all feature of them.
                            example: "/home/user/audio/fft/"

    Note:
         Notice you don't need to specifies file name for saving data.
         The file name should be defined as "fft.bin", "fft-delta.bin" and "fft-delta2.bin" by this function.

    Example: get_delta('/path/to/data/', get_fft, '/path/to/protocol.csv', '/path/to/fft/')
             => /path/to/fft/fft.bin, fft-delta.bin, fft-delta2.bin

    """

    print("Extracting data...")
    data, _ = extractor(df, path_to_data)
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    print("Saving data...")
    save_path = saving_path + "fft.bin"
    save_feature(data, save_path)

    data = data.reshape(data.shape[0], data.shape[1], data.shape[2])

    print("Extracting delta feature...")
    delta_1 = librosa.feature.delta(data, width=5, order=1)
    delta_1 = delta_1.reshape(delta_1.shape[0], delta_1.shape[1], delta_1.shape[2], 1)
    print("Saving data...")
    save_path = saving_path + "fft-delta1.bin"
    save_feature(delta_1, save_path)
    del delta_1

    print("Extracting delta2 feature...")
    delta_2 = librosa.feature.delta(data, width=5, order=2)
    delta_2 = delta_2.reshape(delta_2.shape[0], delta_2.shape[1], delta_2.shape[2], 1)
    print("Saving delta2...")
    save_path = saving_path + "fft-delta2.bin"
    save_feature(delta_2, save_path)

    del data, delta_2
