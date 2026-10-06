import numpy as np

from typing import ClassVar

from eris.core.contracts.base_contract import BaseContract


class UniversalContract(BaseContract):
    """
    This contract validates the data for all encoders.
    """

    contract_name: ClassVar[str] = "universal"

    def validate(self, data):
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        if not data:
            raise ValueError("data cannot be empty")

        for k, v in data.items():
            if not isinstance(v, np.ndarray):
                raise TypeError(f"data[{k}] must be np.ndarray")
