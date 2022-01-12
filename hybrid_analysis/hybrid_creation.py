import json
import os
import shutil
from math import ceil

import numpy as np
from tqdm.auto import tqdm
from sklearn.utils.extmath import randomized_svd

from .utils import Bunch


def create_hybrids(hybrid_dir, cluster_ids, params):
    """
    Make multiple copies of the raw data and add artificial waveforms to each copy with a different
    scaling factor for each dataset
    :param hybrid_dir: Root directory for the hybrid datasets
    :param cluster_ids: Ids of clusters found to be "well-sorted" in the dataset
    :param params:
    :return:
    """
    assert (hybrid_dir / 'raw_data.bin').is_file(), 'Unable to find raw dataset'
    assert (hybrid_dir / 'filtered_data.bin').is_file(), 'Unable to find filtered dataset'
    assert (hybrid_dir / 'output').is_dir(), 'Unable to find spike sorting output'

    # Make copies of the raw datasets
    initialise_datasets(hybrid_dir, params)

    # Add hybrid clusters to the copied datasets
    add_spikes(hybrid_dir, cluster_ids, params)


def initialise_datasets(hybrid_dir, params):
    """
    Create a folder for each hybrid dataset and copy the raw data to each folder
    :param hybrid_dir: Root directory for the hybrid datasets
    :param params:
    """

    data_path = hybrid_dir / 'raw_data.bin'
    assert data_path.is_file()

    meta_path = None
    if 'raw_data.meta' in os.listdir(hybrid_dir):
        meta_path = hybrid_dir / 'raw_data.meta'

    # Folder for storing hybrid datasets
    datasets_dir = hybrid_dir / 'hybrid_datasets'
    if datasets_dir.is_dir():
        for amp in params.scalar_amps_string:
            assert (datasets_dir / f'amp_{amp}' / 'hybrid_data.bin').is_file()
        return
    else:
        os.mkdir(datasets_dir)

    for amp in tqdm(params.scalar_amps_string, desc='Copying Datasets'):
        # Make a folder for each amplitude
        amp_dir = datasets_dir / f'amp_{amp}'
        os.mkdir(amp_dir)
        shutil.copy2(data_path, amp_dir / 'hybrid_data.bin')
        if meta_path:
            shutil.copy2(meta_path, amp_dir / 'hybrid_data.meta')


def get_waveform(time, waveform_shape, channels, data):
    """
    Handles indexing to extract a waveform event from data
    :param time: Event time (centred at peak), int
    :param waveform_shape: 2-length tuple of time offsets for waveform before and after the peak
    :param channels: Waveform channels, numpy array
    :param data: Dataset
    :return: Waveform, numpy array of shape (n_times, n_channels)
    """

    time_start = int(time + waveform_shape[0])
    time_end = int(time + waveform_shape[1])

    return data[time_start:time_end, channels]


def get_spike_events(spike_times, channels, waveform_shape, waveform_width, data):
    """

    :param spike_times:
    :param channels:
    :param waveform_shape:
    :param waveform_width:
    :param data:
    :return:
    """
    waveform_length = waveform_shape[1] - waveform_shape[0] # Total number of times samples

    spike_events = np.zeros((len(spike_times), waveform_length, waveform_width), dtype=np.float32)
    for i in tqdm(range(len(spike_times)), desc='Getting Spikes'):
        spike_events[i] = get_waveform(spike_times[i], waveform_shape, channels, data)

    return spike_events


def denoise_spike_events(spike_events, n_svds):
    """
    Takes in an array of spike waveforms from the same cluster and de-noises them via SVD
    decomposition of the first difference in time
    :param spike_events: Spike waveforms, numpy array of shape (n_spikes, n_times, n_channels)
    :param n_svds: No of svds to use, int
    :return: De-noised spike waveform array with the same shape
    """
    n_spikes, n_times, n_channels = spike_events.shape

    # Take first difference and make the array 2D for SVD decomposition
    spike_events = np.diff(spike_events, axis=1).reshape(spike_events.shape[0], -1)

    svd_comps = randomized_svd(spike_events, n_components=n_svds, random_state=42)
    svd_projs = np.einsum('ij,kj->ik', spike_events, svd_comps[2])

    # No of time samples reduced by 1 after taking first difference
    spike_events = np.matmul(svd_projs, svd_comps[2]).reshape(-1, n_times-1, n_channels)

    # Integrate back
    spike_events = np.cumsum(spike_events, axis=1)

    # Pad with zeros so the number of times samples match input
    spike_events = np.pad(spike_events, ((0,0), (1,0), (0,0)))

    return spike_events


def add_spikes(hybrid_dir, cluster_ids, params):
    """
    Add artificial waveforms to the copies of the raw data
    :param hybrid_dir: Root directory
    :param cluster_ids:
    :param params:
    """

    # Load spike sorting results
    output_path = hybrid_dir / 'output'
    sorting_output = Bunch()
    sorting_output.templates = np.load(output_path / 'templates.npy')
    sorting_output.spike_times = np.load(output_path / 'spike_times.npy')
    sorting_output.spike_templates = np.load(output_path / 'spike_templates.npy')

    # Folder for storing times of inserted hybrid clusters
    (hybrid_dir / 'hybrid_datasets' / 'cluster_times').mkdir(exist_ok=True)

    cluster_amplitudes = np.zeros((len(cluster_ids), 2))

    for i, cluster_id in enumerate(cluster_ids):

        # Re-insert cluster into the hybrid datasets and return the amplitude
        cluster_amplitude = add_cluster_up_down(cluster_id, hybrid_dir, sorting_output, params)
        cluster_amplitudes[i, 0] = cluster_id
        cluster_amplitudes[i, 1] = cluster_amplitude

    np.save(hybrid_dir / 'hybrid_datasets' / 'cluster_amplitudes.npy', cluster_amplitudes)


def add_cluster_up_down(cluster_id, hybrid_dir, sorting_output, params):
    """
    Extract the spike events for a specific cluster and re-insert it above and below its initial
    position with some time offsets
    :param cluster_id: CLuster to re-insert, int
    :param hybrid_dir: Path to hybrid folder
    :param sorting_output: Output of initial spike-sorting, must have templates, spike times and
                            spike templates
    :param params:
    :return:
    """
    # Filtered data for extracting waveforms
    data_filtered = np.memmap(hybrid_dir / 'filtered_data.bin', dtype='float32', mode='r').reshape(-1,385)

    # New hybrid datasets
    hybrid_paths = [hybrid_dir / 'hybrid_datasets' / f'amp_{amp}' / 'hybrid_data.bin'
                    for amp in params.scalar_amps_string]
    hybrid_datasets = [np.memmap(path, dtype=np.int16).reshape(-1, params.n_channels)
                       for path in hybrid_paths]

    cluster_times_dir = hybrid_dir / 'hybrid_datasets' / 'cluster_times'

    main_channel = np.argmax(np.abs(sorting_output.templates[cluster_id, params.template_peak]))

    cluster_channels = get_nearest_channels(params.probe, params.waveform_width)[main_channel]

    event_times = sorting_output.spike_times[sorting_output.spike_templates == cluster_id]

    # Extract spike waveforms and denoise them
    spike_events = get_spike_events(event_times, cluster_channels, params.waveform_shape,
                                    params.waveform_width, data_filtered)
    spike_events = denoise_spike_events(spike_events, params.n_svds)

    cluster_amplitude = np.median(np.max(np.abs(spike_events[:, -params.waveform_shape[0]]), axis=1))

    n_samples = data_filtered.shape[0]
    n_spikes = len(event_times)

    batch_size = params.batch_size_hybrid

    n_batches = ceil(n_samples / batch_size)

    channel_shifts = [-params.channel_shift, params.channel_shift]

    # Modify the relative shifts of the clusters if they're too close to one end of the probe
    if min(cluster_channels) < params.channel_shift:
        channel_shifts = [params.channel_shift, 2*params.channel_shift]

    if max(cluster_channels) > params.n_channels - 1 - params.channel_shift:
        channel_shifts = [-params.channel_shift, -2*params.channel_shift]

    all_times = []
    all_shifts = []
    all_event_ids = []

    for i, shift in enumerate(channel_shifts):
        hybrid_times = event_times + (i+1) * params.time_shift

        # Make sure all of the new times are within range of the dataset
        valid_events = hybrid_times < n_samples - params.waveform_length
        hybrid_times = hybrid_times[valid_events]

        # Save times of inserted cluster
        np.save(cluster_times_dir / f'cluster_{cluster_id}_channel_{main_channel + shift}', hybrid_times)

        all_times.append(hybrid_times)
        all_shifts.append((np.ones(n_spikes, dtype='int') * shift)[valid_events])
        all_event_ids.append(np.arange(n_spikes)[valid_events])

    all_times = np.concatenate(all_times)
    all_shifts = np.concatenate(all_shifts)
    all_event_ids = np.concatenate(all_event_ids)


    for i in tqdm(range(n_batches), desc=f'Adding Batches for cluster {cluster_id}'):

        batch_idx = np.logical_and(all_times >= i*batch_size, all_times < (i+1)*batch_size)
        batch_times = all_times[batch_idx] - i*batch_size
        batch_shifts = all_shifts[batch_idx]
        batch_event_ids = all_event_ids[batch_idx]

        batch_data = add_spikes_batch(spike_events, batch_times, batch_shifts, batch_event_ids,
                                      cluster_channels, batch_size, params)

        if i == 0:
            batch_data = batch_data[params.waveform_length:]
        if i == n_batches - 1:
            upper_buffer = params.waveform_length + (i+1) * batch_size - n_samples
            batch_data = batch_data[:-upper_buffer]
            #needs to be modified when batch_size isn't a divisor of n_samples

        for j in range(len(params.scalar_amps)):
            hybrid_datasets[j][max(i * batch_size - params.waveform_length, 0):
                               min((i+1) * batch_size + params.waveform_length, n_samples)] += \
                (batch_data * params.scalar_amps[j]).astype(np.int16)

    return cluster_amplitude


def add_spikes_batch(spike_events, batch_times, batch_shifts, batch_event_ids, base_channels,
                     batch_size, params):
    """
    For a batch of data, creates a template of all spike events that need to be added in memory
    so the entire batch can be added to the hybrid dataset at once
    Done this way to minimise memmap calls to the disk
    :param spike_events: De-noised spike waveforms
    :param batch_times: Times of spike events within the batch
    :param batch_shifts: Channel shifts for inserted spike events
    :param batch_event_ids: Waveform ids of spike events
    :param base_channels: Channels of original cluster
    :param batch_size: Size of batch, int
    :param params:
    :return:
    """
    buffer = params.waveform_length
    batch_data = np.zeros((batch_size + 2*buffer, params.n_channels))
    for i in range(len(batch_times)):
        start_time = int(batch_times[i] + params.waveform_shape[0] + buffer)
        end_time = int(batch_times[i] + params.waveform_shape[1] + buffer)
        batch_data[start_time:end_time, base_channels + batch_shifts[i]] += spike_events[batch_event_ids[i]]
    return batch_data



def get_nearest_channels(probe, k_nearest = None):
    """
    For each channel returns its nearest neighbours
    :param probe: Probe object with xc and yc attributes for the channel coordinates
    :param k_nearest: (Optional) Number of nearest neighbours required, defaults to n_channels
    :return: Numpy array of shape (n_channels, k_nearest)
    """

    channel_distances = (probe.xc - probe.xc.reshape(-1,1)) ** 2 + \
                        (probe.yc - probe.yc.reshape(-1,1)) ** 2

    if not k_nearest:
        return np.argsort(channel_distances)

    return np.argsort(channel_distances)[:,:k_nearest]
