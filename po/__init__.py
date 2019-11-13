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
        return len(next(iter(self._data.values())))

    @property
    def columns(self):
        return list(self._data.keys())

    @columns.setter
    def columns(self, col_names):
        if not isinstance(col_names, list):
            raise TypeError(
                f"`col_names` should be of type list- got {type(col_names)} instead."
            )
        if len(col_names) != len(self._data.keys()):
            raise ValueError(
                "`col_names` should be of length {len(self._data.keys())}- got {len(col_names) instead.}"
            )
        if not all((isinstance(x, str) for x in col_names)):
            raise TypeError("All column names should be of type `str`.")
        if len(set(col_names)) < len(col_names):
            raise ValueError("`col_names` should not contain duplicates.")

        self._data = dict(zip(col_names, self._data.values()))

    @property
    def shape(self):
        return (self.__len__(), len(self._data))

    def _repr_html_(self):
        """
            A nice visual representation of the df for Jupyter Notebooks
        """
        pass

    @property
    def values(self):
        return np.stack(list(self._data.values()))

    @property
    def dtypes(self):
        DTYPES = {"O": "string", "i": "int", "f": "float", "b": "bool"}
        return DataFrame(
            {
                "column_name": np.array(list(self._data.keys())),
                "dtype": np.array([DTYPES[v.dtype.kind] for v in self._data.values()]),
            }
        )

    def __getitem__(self, index: str):
        if isinstance(index, str):
            return self._get_single_column(index)

    def _get_single_column(self, index: str):
        if not index in self.columns:
            raise ValueError(f"`{index}` not found in columns: {self.columns}")
        return self._data[index]
