import numpy as np


class DataFrame:
    def __init__(self, data):
        """
            A DataFrame has 2 dimensional heterogenous data.
        """
        self._check_input_types(data)
        self._check_input_lengths(data)
        self._data = self._convert_unicode_to_object(data)

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

    def _check_input_lengths(self, data):
        for i, value in enumerate(data.values()):
            if i == 0:
                length = len(value)
            elif len(value) != length:
                raise ValueError(
                    f"values of `data` should be of same length. Got {len(value)} and {length}"
                )

    def _convert_unicode_to_object(self, data):
        """
            a = np.array(['apple', 'ball', 'cat'])
            a[1] = None
            a = np.array(['apple', 'None', 'cat'])
        """
        converted_data = {}

        for k, v in data.items():
            if v.dtype.kind == "U":
                converted_data[k] = v.astype("object")
            else:
                converted_data[k] = v
        return converted_data

    def __len__(self):
        for value in self._data.values():
            return len(value)
