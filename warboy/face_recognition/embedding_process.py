from typing import Dict, List

import numpy as np


class face_recognition_embedding_decoder:
    def __init__(self):
        pass

    def __call__(
        self, outputs: List[np.ndarray], contexts: Dict[str, float], img: np.ndarray
    ) -> np.ndarray:
        return outputs
