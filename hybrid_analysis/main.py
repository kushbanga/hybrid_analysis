import os
from pathlib import Path
import shutil

from one.api import OneAlyx

# from pykilosort import run, add_default_handler

# from .preprocess import preprocess_raw_data
from .params import Params
from .utils import np1_probe, change_params_path
from .hybrid_creation import create_hybrids


ROOT_DIR = Path(r'D:/hybrid_datasets')
IBL_DIR = Path(r'D:/ibl_data')
PHY_DIR = Path(r'C:\Users\Experiment\Documents\Kush\ks_output_phy')


def setup_downloaded(pid, ephys_data_path, **kwargs):
    """
    For a given pid with downloaded data, save a raw snippet, a filtered snippet and other useful
    information, then run spike sorting
    :param pid: Probe ID, string
    :param ephys_data_path: Path to data, Pathlib path
    """
    params = Params(**kwargs)

    hybrid_dir = ROOT_DIR / pid
    if not os.path.isdir(hybrid_dir):
        os.mkdir(hybrid_dir)

    # # Saves raw snippet and filtered snippet in hybrid_dir
    # preprocess_raw_data(ephys_data_path, hybrid_dir, params)

    # Run spike sorting
    data_path = hybrid_dir / 'raw_data.bin'
    # ctx = run(data_path, probe=np1_probe())
    # del ctx

    # Clean up temporary kilosort files
    shutil.rmtree(hybrid_dir / '.kilosort')

    # Copy output to Phy folder to look for good clusters
    shutil.copyfile(data_path, PHY_DIR)
    shutil.copytree(hybrid_dir / 'output', PHY_DIR)

    # Fix path to raw data in Phy params.py file
    change_params_path(PHY_DIR)


def setup(pid, **kwargs):
    """
    For a given probe id, downloads data, saves a raw snippet, a filtered snippet and other useful
    information, then runs spikes sorting
    :param pid: Probe ID, string
    """

    params = Params(**kwargs)

    one = OneAlyx(cache_dir=IBL_DIR)

    eid, probe_name = one.pid2eid(pid)
    probe_number = probe_name[-1]

    # Get relevant files
    all_files = one.list_datasets(eid)
    ephys_data_files = [file for file in all_files if
                        file.endswith(('ap.cbin', 'ap.ch', 'ap.meta'))
                        and f'imec{probe_number}' in file]

    # Download IBL Raw Data
    ephys_data_paths = []
    for file in ephys_data_files:
        file_path = one.load_dataset(eid, file, download_only=True)
        ephys_data_paths.append(file_path)

    ephys_data_path = [file for file in ephys_data_paths if file.suffix == '.cbin'][0]

    hybrid_dir = ROOT_DIR / pid
    if not os.path.isdir(hybrid_dir):
        os.mkdir(hybrid_dir)

    # # Saves raw snippet and filtered snippet in hybrid_dir
    # preprocess_raw_data(ephys_data_path, hybrid_dir, params)

    # Delete downloaded data
    shutil.rmtree(ephys_data_path.parent)

    # Run spike sorting
    data_path = hybrid_dir / 'raw_data.bin'
    # ctx = run(data_path, probe=np1_probe())
    # del ctx

    # Clean up temporary kilosort files
    shutil.rmtree(hybrid_dir / '.kilosort')

    # Copy output to Phy folder to look for good clusters
    shutil.copyfile(data_path, PHY_DIR)
    shutil.copytree(hybrid_dir / 'output', PHY_DIR)

    # Fix path to raw data in Phy params.py file
    change_params_path(PHY_DIR)


def run_hybrids(hybrid_dir, cluster_ids, **kwargs):
    """

    :param pid: Probe id, string
    :param cluster_ids: Clusters to be re-inserted that were "well-sorted", list of integers
    """
    params = Params(**kwargs)

    # # Delete files in Phy folder
    # os.remove(PHY_DIR / 'raw_data.bin')
    # shutil.rmtree(PHY_DIR / 'output')
    #
    # hybrid_dir = ROOT_DIR / pid
    output_dir = hybrid_dir / 'output'

    # Create folders for hybrid datasets and copy the raw data to each folder
    create_hybrids(hybrid_dir, cluster_ids, params)
