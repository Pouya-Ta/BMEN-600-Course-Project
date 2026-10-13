"""Fixed Welch settings and channel-level band powers, independent of evaluation."""
import numpy as np
from scipy.signal import welch
from shared import CHANNELS

BANDS = {"delta": (1., 4.), "theta": (4., 8.), "alpha": (8., 13.), "beta": (13., 30.)}


def feature_names():
    return [f"{channel}__{kind}__{band}" for channel in CHANNELS
            for kind in ["log10_absolute", "relative"] for band in BANDS]


def integrate_psd(frequency, psd, lower, upper):
    """Trapezoidal integral of the linearly interpolated PSD over [lower, upper].

    Adjacent bands share a zero-measure endpoint, not overlapping frequency bins.
    """
    if lower < frequency[0] or upper > frequency[-1]:
        raise ValueError("Requested integration range exceeds the PSD")
    grid = np.r_[lower, frequency[(frequency > lower) & (frequency < upper)], upper]
    curves = np.stack([np.interp(grid, frequency, row) for row in psd])
    return np.trapezoid(curves, grid, axis=-1)


def extract(epoch_uv, fs, config):
    nperseg = int(round(config["welch_seconds"] * fs))
    if nperseg > epoch_uv.shape[-1] or nperseg < 2:
        raise ValueError("Welch subwindow must fit inside the shared epoch")
    frequency, psd = welch(epoch_uv, fs=fs, window="hann", nperseg=nperseg,
                           noverlap=nperseg // 2, nfft=nperseg, detrend="constant",
                           scaling="density", average="mean", axis=-1)
    power = np.stack([integrate_psd(frequency, psd, *bounds) for bounds in BANDS.values()], axis=-1)
    denominator = integrate_psd(frequency, psd, 1., 30.)[:, None]
    floor = config["power_floor_uv2"]
    logarithm = np.log10(np.maximum(power, floor))
    relative = power / np.maximum(denominator, floor)
    vector = np.concatenate([logarithm, relative], axis=-1).ravel()
    if not np.isfinite(vector).all() or len(vector) != len(CHANNELS) * 8:
        raise ValueError("Invalid feature dimensions/values")
    if not np.allclose(relative.sum(axis=-1), 1., atol=1e-8):
        raise ValueError("Band powers do not partition the 1–30 Hz denominator")
    keep = (frequency >= .5) & (frequency <= 45.)
    return vector, frequency[keep], psd[:, keep], int((power < floor).sum())
