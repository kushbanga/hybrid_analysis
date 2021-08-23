from pydantic import BaseModel, Field

class Params(BaseModel):

    start_time: int = Field(1000, description='Start time of data snippet used')
    end_time: int = Field(2200, description='End time of data snippet used')

    sample_rate: int = Field(30000, description='Sample Rate')

    fshigh: float = Field(300.0, description="High pass filter frequency")

