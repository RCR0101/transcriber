from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _patch_torchaudio_compat():
    """Shim for DeepFilterNet's broken torchaudio.backend.common import on torchaudio >=2.1."""
    if "torchaudio.backend.common" in sys.modules:
        return

    class AudioMetaData:
        def __init__(self, sample_rate=0, num_frames=0, num_channels=0,
                     bits_per_sample=0, encoding=""):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    mod = types.ModuleType("torchaudio.backend.common")
    mod.AudioMetaData = AudioMetaData
    sys.modules["torchaudio.backend.common"] = mod

    if "torchaudio.backend" not in sys.modules:
        backend_mod = types.ModuleType("torchaudio.backend")
        backend_mod.common = mod
        sys.modules["torchaudio.backend"] = backend_mod


def _run_deepfilter(audio: np.ndarray, sr: int) -> np.ndarray:
    import torch
    _patch_torchaudio_compat()
    from df.enhance import enhance, init_df

    model, df_state, _ = init_df()
    df_sr = df_state.sr()

    if sr != df_sr:
        import torchaudio.functional as F
        tensor = torch.from_numpy(audio).float().unsqueeze(0)
        tensor = F.resample(tensor, sr, df_sr)
    else:
        tensor = torch.from_numpy(audio).float().unsqueeze(0)

    enhanced = enhance(model, df_state, tensor)

    if sr != df_sr:
        import torchaudio.functional as F
        enhanced = F.resample(enhanced, df_sr, sr)

    return enhanced.squeeze(0).numpy()


def _run_noisereduce(audio: np.ndarray, sr: int) -> np.ndarray:
    import noisereduce as nr
    return nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)


def denoise_file(input_path: str | Path, output_path: str | Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info(f"Loading audio from {input_path}")
    audio, sr = sf.read(str(input_path), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    logger.info("Stage 1: DeepFilterNet noise suppression")
    audio = _run_deepfilter(audio, sr)

    logger.info("Stage 2: Spectral gating for residual noise")
    audio = _run_noisereduce(audio, sr)

    sf.write(str(output_path), audio, sr)
    logger.info(f"Denoised audio saved to {output_path}")
