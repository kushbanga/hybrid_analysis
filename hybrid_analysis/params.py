from pydantic import BaseModel, Field

class Params(BaseModel):

    start_time: int = Field(1000, description='Start time of data snippet used')
    end_time: int = Field(2200, description='End time of data snippet used')

    sample_rate: int = Field(30000, description='Sample Rate')

    fshigh: float = Field(300.0, description="High pass filter frequency")

    waveform_shape = Field((-21, 40), description='Time samples either side of spike event')

    waveform_width: int = Field(20, description='No of channels per spike event')

    n_svds: int = Field(9, description='No of SVD components used in waveform de-noising')

    @property
    def waveform_length(self):
        return self.waveform_shape[1] - self.waveform_shape[0]

