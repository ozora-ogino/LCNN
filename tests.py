# pylint: disable=W0613

import numpy as np
import pandas as pd

from src.feature import _calc_cqt, _calc_stft, _extract_label, calc_cqt, calc_stft


class Test_calcStft:
    """Test for `_calc_stft`"""

    def test_shape(self, mocker):
        def _dummy_librosa_load(path):
            return ([x for x in range(100000)], 16000)

        mocker.patch("librosa.load", _dummy_librosa_load)
        assert len(_calc_stft("dummy_path.flac").shape) == 3

    def test_is_ndarray(self, mocker):
        def _dummy_librosa_load(path):
            return ([x for x in range(100000)], 16000)

        mocker.patch("librosa.load", _dummy_librosa_load)
        assert type(_calc_stft("dummy_path.flac")) == np.ndarray


class TestCalcStft:
    """Test for `calc_stft`"""

    def test_shape(self, mocker):
        dummy = {
            "protocol_df": pd.DataFrame({"utt_id": ["dummy-1", "dummy-2"]}),
            "path": "dummy_path",
        }

        mocker.patch("src.feature._calc_stft", lambda x: np.random.randn(100, 100, 1))
        mocker.patch("src.feature._extract_label", lambda x: np.random.randint(0, 2, 2))
        data, _ = calc_stft(**dummy)
        assert len(data.shape) == 4


class Test_calcCqt:
    """Test for `_calc_cqt`"""

    def test_shape(self, mocker):
        def _dummy_librosa_load(path):
            return ([x for x in range(100000)], 16000)

        mocker.patch("librosa.load", _dummy_librosa_load)
        assert len(_calc_cqt("dummy_path.flac").shape) == 2

    def test_is_ndarray(self, mocker):
        def _dummy_librosa_load(path):
            return ([x for x in range(100000)], 16000)

        mocker.patch("librosa.load", _dummy_librosa_load)
        assert type(_calc_cqt("dummy_path.flac")) == np.ndarray


class TestCalcCqt:
    """Test for `calc_cqt`"""

    def test_shape(self, mocker):
        dummy = {
            "protocol_df": pd.DataFrame({"utt_id": ["dummy-1", "dummy-2"]}),
            "path": "dummy_path",
        }

        mocker.patch("src.feature._calc_cqt", lambda x: np.random.randn(100, 100))
        mocker.patch("src.feature._extract_label", lambda x: np.random.randint(0, 2, 2))
        data, _ = calc_cqt(**dummy)
        assert len(data.shape) == 4


class TestExtractsLabel:
    """Test for `_extract_label`"""

    def test_dtype_is_int(self):
        dummy = pd.DataFrame({"key": ["bonafide", "spoof", "bonafide"]})

        assert _extract_label(dummy).dtype == np.dtype("int")
