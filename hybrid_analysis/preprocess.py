import numpy as np
import cupy as cp
from tqdm import tqdm

from pykilosort.preprocess import gpufilter
from phylib.io.traces import get_ephys_reader


# Size (in seconds) of batches for processing
BATCH_TIME = 3

# Size of buffer (in samples) for filtering
BUFFER_SIZE = 100


def preprocess_raw_data(data_path, output_dir, params):
    """
    Creates a raw snippet and filtered snippet and stores these in output_dir as well as the
    standard deviations for each channel
    :param data_path: Path to raw data
    :param output_dir: Path to directory for storing output
    :param params: Params object
    """
    batch_size = BATCH_TIME * params.sample_rate
    start_ind = params.start_time * params.sample_rate

    raw_output_path = output_dir / 'raw_data.bin'
    filtered_output_path = output_dir / 'filtered_data.bin'

    with open(raw_output_path, 'wb') as fwr, open(filtered_output_path, 'wb') as fwf:

        raw_data = get_ephys_reader(data_path, n_channels=385, dtype=np.int16,
                                    sample_rate=params.sample_rate)
        time_length = (params.end_time - params.start_time) * params.sample_rate
        n_batches = time_length // batch_size
        channel_devs = np.zeros((n_batches, 385))

        for i in tqdm(range(n_batches)):

            # Buffers to handle edge artefacts in filtering
            # Special care for first and last batches
            lower_buffer = 0 if i == 0 else lower_buffer = BUFFER_SIZE
            upper_buffer = 0 if i == n_batches - 1 else upper_buffer = BUFFER_SIZE

            lower_index = start_ind + batch_size*i - lower_buffer
            upper_index = start_ind + batch_size*(i+1) + upper_buffer
            sub_data = raw_data[lower_index: upper_index]

            filtered_data = gpufilter(cp.array(sub_data), fshigh=150, fs=30000)
            filtered_data_cpu = cp.asnumpy(filtered_data)

            sub_data = sub_data[lower_buffer: sub_data.shape[0] - upper_buffer]
            assert sub_data.shape[0] == batch_size
            sub_data.tofile(fwr)

            filtered_data_cpu = filtered_data_cpu[lower_buffer:
                                                  filtered_data_cpu.shape[0] - upper_buffer]
            assert filtered_data_cpu.shape[0] == batch_size
            channel_devs[i] = np.std(filtered_data_cpu, axis=0)
            filtered_data_cpu.tofile(fwf)

        channel_devs = np.median(channel_devs, axis=0)
        np.save(output_dir / 'channel_devs.npy', channel_devs)
