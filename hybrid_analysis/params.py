from pydantic import BaseModel, Field
import numpy as np


class Probe(BaseModel):

    xc: np.ndarray
    yc: np.ndarray

    class Config:
        arbitrary_types_allowed = True


def np1_probe_constructor():

    kwargs = {
        'xc': np.tile(np.array([43, 11, 59, 27], dtype='float'), 96),
        'yc': np.repeat(np.arange(20, 3841, 20.), 2),
    }
    return Probe(**kwargs)


class Params(BaseModel):

    start_time: int = Field(1300, description='Start time of data snippet used')
    end_time: int = Field(2500, description='End time of data snippet used')

    sample_rate: int = Field(30000, description='Sample Rate')

    fshigh: float = Field(300.0, description="High pass filter frequency")

    waveform_shape = Field((-21, 40), description='Time samples either side of spike event')

    waveform_width: int = Field(20, description='No of channels per spike event')

    n_svds: int = Field(9, description='No of SVD components used in waveform de-noising')

    template_peak: int = Field(41, description='Index at which the template peak is aligned')

    # TODO: Pass a numpy array of scalar amps directly as a field
    min_amp: float = Field(0.1, description='Minimum scalar amplitude')
    max_amp: float = Field(1., description='Maximum Scalar Amplitude')
    amp_step: float = Field(0.1, description='Step size between different scalar amplitudes')

    probe = Field(np1_probe_constructor(), description='Probe object - must have attributes xc and'
                                        'yc that store the x and y co-ordinates of the channels')

    n_channels: int = Field(385, description='No of channels in the dataset')

    batch_size_hybrid: int = Field(100000, description='No of times samples for the batches when'
                                                       'hybrid spikes are added')

    channel_shift: int = Field(24, description='No of channels to shift a cluster up/down by when'
                                               'adding to a hybrid dataset')

    time_shift: int = Field(100, description='No of time samples to shift a cluster forward by'
                                             'when adding to a hybrid dataset')

    @property
    def waveform_length(self):
        return self.waveform_shape[1] - self.waveform_shape[0]

    @property
    def scalar_amps(self):
        return np.arange(self.min_amp, self.max_amp + 0.5 * self.amp_step, self.amp_step)

    @property
    def scalar_amps_string(self):
        scalar_amps = self.scalar_amps
        return ['{0:.2f}'.format(scalar_amp) for scalar_amp in scalar_amps]
