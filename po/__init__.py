import numpy as np


class DataFrame:
    def __init__(self, data):
        """
            A DataFrame has 2 dimensional heterogenous data.
        """
        self._check_input_types(data)
        pass

    def _check_input_types(self, data):
        if not isinstance(data, dict):
            raise TypeError(f"`data` should be of type dict- got {type(data)} instead.")

        for k, v in data.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"All keys of `data` should be of type str. Got {type(k)} for {k}"
                )

            if not isinstance(v, np.ndarray):
                raise TypeError(
                    f"All values of `data` should be of type np.ndarray. Got {type(v)} for {v}"
                )

            if v.ndim != 1:
                raise TypeError(
                    f"All values of `data` should be 1 dimensional. Got {v.ndim} for {v}"
                )
