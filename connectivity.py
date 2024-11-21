import numpy as np
from scipy import signal, stats
from typing import Optional, Literal, Tuple
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
import scipy.spatial.distance as distance


def calculate_connectivity(
        eeg_data: np.ndarray,
        connectivity_measure: Literal['pli', 'coherence', 'granger', 'mi', 'transfer_entropy', 'phase_sync'] = 'phase_sync',
        sampling_rate: Optional[float] = None
) -> np.ndarray:
    """
    Compute connectivity between EEG channel time series.

    Parameters:
    -----------
    eeg_data : numpy.ndarray
        2D array of EEG data with shape (num_channels, num_points)
    connectivity_measure : str, optional
        Type of connectivity measure
    sampling_rate : float, optional
        Sampling rate of the EEG data

    Returns:
    --------
    connectivity_matrix : numpy.ndarray
        Symmetric matrix of connectivity values between channels
    """
    num_channels, num_points = eeg_data.shape
    connectivity_matrix = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        # for the time series of each channel
        channel_i_data = eeg_data[i, :]

        for j in range(i + 1, num_channels):
            # vs every other channel
            channel_j_data = eeg_data[j, :]

            # Select connectivity measure and compute it between these channels
            if connectivity_measure == 'coherence':
                if sampling_rate is None:
                    raise ValueError("Sampling rate required for coherence")
                connectivity_value = compute_coherence(channel_i_data, channel_j_data, sampling_rate)
            elif connectivity_measure == 'phase_sync':
                connectivity_value = compute_phase_synchronization(channel_i_data, channel_j_data)
            elif connectivity_measure == 'granger':
                connectivity_value = compute_granger_causality(channel_i_data, channel_j_data)
            elif connectivity_measure == 'mi':
                connectivity_value = compute_mutual_information(channel_i_data, channel_j_data)
            elif connectivity_measure == 'pli':
                connectivity_value = compute_pli(channel_i_data, channel_j_data)

            else:
                raise ValueError("Unsupported connectivity measure")

            connectivity_matrix[i, j] = connectivity_value
            connectivity_matrix[j, i] = connectivity_value

    return connectivity_matrix


def compute_distance_weighted_correlation(
        signal1: np.ndarray,
        signal2: np.ndarray,
        electrode_locations: np.ndarray,
        distance_decay_param: float = 0.5
) -> np.ndarray:
    """
    Calculate distance-weighted correlation between two EEG signals.

    Parameters:
    -----------
    signal1 : numpy.ndarray
        Time series of the first electrode signal
    signal2 : numpy.ndarray
        Time series of the second electrode signal
    electrode_locations : numpy.ndarray
        2D array of (x, y, z) coordinates for electrodes
    distance_decay_param : float, optional
        Controls how quickly correlation decays with distance
        Higher values = faster decay

    Returns:
    --------
    float
        Distance-weighted correlation coefficient
    """
    # Calculate Euclidean distance between electrodes
    inter_electrode_distance = distance.euclidean(
        electrode_locations[0],
        electrode_locations[1]
    )

    # Calculate standard Pearson correlation
    raw_correlation = np.corrcoef(signal1, signal2)[0, 1]

    # Apply distance-based exponential decay
    distance_weight = np.exp(-distance_decay_param * inter_electrode_distance)

    # Weight the correlation
    weighted_correlation = raw_correlation * distance_weight

    return weighted_correlation


def compute_pli(
        signal1: np.ndarray,
        signal2: np.ndarray
) -> float:
    """Compute Phase Lag Index (PLI)"""
    phase1 = np.angle(signal.hilbert(signal1))
    phase2 = np.angle(signal.hilbert(signal2))
    return np.abs(np.mean(np.sign(phase1 - phase2)))


def compute_coherence(
        signal1: np.ndarray,
        signal2: np.ndarray,
        sampling_rate: float
) -> float:
    """Compute signal coherence"""

    f, Cxy = signal.coherence(signal1, signal2, fs=sampling_rate)
    return np.mean(Cxy)


def compute_granger_causality(
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int = 5
) -> float:
    """Compute Granger Causality"""


    # Prepare data as a DataFrame
    data = np.column_stack([x, y])
    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    # Return the p-value for the lowest lag
    return 1 - results[1][0]['ssr_ftest'][1]


def compute_mutual_information(
        x: np.ndarray,
        y: np.ndarray,
        bins: int = 10
) -> float:
    """Compute Mutual Information"""
    # Joint probability distribution
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    c_x, _ = np.histogram(x, bins=bins)
    c_y, _ = np.histogram(y, bins=bins)

    # Normalize
    c_xy /= float(np.sum(c_xy))
    c_x /= float(np.sum(c_x))
    c_y /= float(np.sum(c_y))

    # Compute MI
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if c_xy[i, j] > 0:
                mi += c_xy[i, j] * np.log2(c_xy[i, j] / (c_x[i] * c_y[j]))

    return mi


def compute_transfer_entropy(
        x: np.ndarray,
        y: np.ndarray,
        lag: int = 1
) -> float:
    """Compute Transfer Entropy"""
    # Embed time series
    x_lagged = x[:-lag]
    y_past = y[:-lag]
    y_future = y[lag:]

    # Compute conditional probabilities
    _, p_x_ypast = pearsonr(x_lagged, y_past)
    _, p_y_future_x_ypast = pearsonr(y_future, np.column_stack([x_lagged, y_past]))

    return np.log(p_y_future_x_ypast / p_x_ypast)


def compute_phase_synchronization(
        x: np.ndarray,
        y: np.ndarray
) -> float:
    """Compute Phase Synchronization"""
    # Hilbert transform to get instantaneous phases
    x_phase = np.angle(signal.hilbert(x))
    y_phase = np.angle(signal.hilbert(y))

    # Compute phase difference
    phase_diff = np.abs(x_phase - y_phase)

    # Kuramoto order parameter
    return np.abs(np.mean(np.exp(1j * phase_diff)))

