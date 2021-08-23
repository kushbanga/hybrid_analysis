import numpy as np
from pykilosort import Bunch


def np1_probe():
    probe = Bunch()
    probe.NchanTOT = 385
    probe.chanMap = np.arange(384)
    probe.xc = np.tile(np.array([43,11,59,27], dtype='float'), 96)
    probe.yc = np.repeat(np.arange(20, 3841, 20.), 2)
    probe.kcoords = np.zeros(384)
    return probe


def change_params_path(phy_dir):
    """
    Modifies path to raw data in params.py file so it is correct after copying
    :param phy_dir: Path to directory storing copied spike sorting results
    """
    # Path to params.py file
    params_path = phy_dir / 'output' / 'params.py'

    # Load content
    content = params_path.read_text()
    content = content.split(sep='\n')

    # Modify file path
    assert content[0][:8] == 'dat_path'
    content[0] = f'dat_path = r"{str(phy_dir / "raw_data.bin")}"'
    content = '\n'.join(content)

    # Write changes to params.py file
    with open(params_path, 'w') as f:
        f.truncate(0)
        f.write(content)
