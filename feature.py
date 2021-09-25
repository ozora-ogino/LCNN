import pickle
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm


def _preEmphasis(wave: np.ndarray, p=0.97) -> np.ndarray:
    """Pre-Emphasis"""
    return scipy.signal.lfilter([1.0, -p], 1, wave)


def _calc_stft(path: str) -> np.ndarray:
    """Calculate STFT with librosa.

    Args:
        path (str): Path to audio file

    Returns:
        np.ndarray: A STFT spectrogram.
    """
    wave, sr = librosa.load(path)
    wave = _preEmphasis(wave)
    steps = int(len(wave) * 0.0081)
    # calculate STFT
    stft = librosa.stft(wave, n_fft=sr, win_length=1700, hop_length=steps, window="blackman")
    amp_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    amp_db = amp_db[:800, :].astype("float32")
    return amp_db[..., np.newaxis]


def calc_stft(protocol_df: pd.DataFrame, path: str) -> Tuple[np.ndarray, np.ndarray]:
    """

    This function extracts spectrograms from raw audio data by using FFT.

    Args:
     protocol_df(pd.DataFrame): ASVspoof2019 protocol.
     path(str): Path to ASVSpoof2019

    Returns:
     data: spectrograms that have 4 dimentions like (n_samples, height, width, 1)
     label: 0 = Genuine, 1 = Spoof
    """

    data = []
    for audio in tqdm(protocol_df["utt_id"]):
        file = path + audio + ".flac"
        # Calculate STFT
        stft_spec = _calc_stft(file)
        data.append(stft_spec)

    # Extract labels from protocol
    labels = _extract_label(protocol_df)

    return np.array(data), labels


def _calc_cqt(path: str) -> np.ndarray:
    """Calculating CQT spectrogram

    Args:
        path (str): Path to audio file.

    Returns:
        np.ndarray: A CQT spectrogram.
    """
    y, sr = librosa.load(path)
    y = _preEmphasis(y)
    cqt_spec = librosa.core.cqt(y, sr=sr)
    cq_db = librosa.amplitude_to_db(np.abs(cqt_spec))  # Amplitude to dB.
    return cq_db


def calc_cqt(protocol_df: pd.DataFrame, path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate spectrograms from raw audio data by using CQT.

    Please refer to `calc_stft` for arguments and returns
    They are almost same.

    """

    samples = protocol_df["utt_id"]
    max_width = 200  # for resizing cqt spectrogram.

    for i, sample in enumerate(tqdm(samples)):
        full_path = path + sample + ".flac"
        # Calculate CQT spectrogram
        cqt_spec = _calc_cqt(full_path)

        height = cqt_spec.shape[0]
        if i == 0:
            resized_data = np.zeros((len(protocol_df), height, max_width))

        # Truncate
        if max_width <= cqt_spec.shape[1]:
            cqt_spec = cqt_spec[:, :max_width]
        else:
            # Zero padding
            diff = max_width - cqt_spec.shape[1]
            zeros = np.zeros((height, diff))
            cqt_spec = np.concatenate([cqt_spec, zeros], 1)

        resized_data[i] = np.float32(cqt_spec)

    # Extract labels from protocol
    labels = _extract_label(protocol_df)

    return resized_data[..., np.newaxis], labels


def _extract_label(protocol: pd.DataFrame) -> np.ndarray:
    """Extract labels from ASVSpoof2019 protocol

    Args:
        protocol (pd.DataFrame): ASVSpoof2019 protocol

    Returns:
        np.ndarray: Labels.
    """
    labels = np.ones(len(protocol))
    labels[protocol["key"] == "bonafide"] = 0
    return labels.astype(int)


def save_feature(feature: np.ndarray, path: str):
    """Save spectrograms as a binary file.

    Args:
        feature (np.ndarray): Spectrograms with 4 dimensional shape like (n_samples, height, width, 1)
        path (str): Path for saving.
    """
    with open(path, "wb") as web:
        pickle.dump(feature, web, protocol=4)
