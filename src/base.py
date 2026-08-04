from dataclasses import dataclass,field
import numpy as np


@dataclass
class CVResult:
    """it will return uniform result for every 
    Uniform return type for every module.

    overlay : image for human viewing (RGB, uint8), or None
    data    : structured results — this is what the pipeline consumes
    meta    : provenance — which model, what settings, how long
    """

    overlay : np.ndarray | None=None
    data:dict= field(default_factory=dict)
    meta:dict= field(default_factory=dict)