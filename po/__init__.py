import numpy as np
from prettytable import PrettyTable
from typing import List, Union

DTYPES = {"O": "string", "i": "int", "f": "float", "b": "bool"}


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

    def __repr__(self):
        x = PrettyTable()
        for k, v in self._data.items():
            x.add_column(k, v)
        return x.get_string()

    @property
    def values(self):
        return np.stack(list(self._data.values()))

    @property
    def dtypes(self):
        return DataFrame(
            {
                "column_name": np.array(list(self._data.keys())),
                "dtype": np.array([DTYPES[v.dtype.kind] for v in self._data.values()]),
            }
        )

    def __getitem__(self, index: Union[str, List[str]]):
        if isinstance(index, str):
            return self._get_single_column(index)
        elif isinstance(index, list):
            return self._get_multiple_columns(index)
        elif isinstance(index, DataFrame):
            return self._get_subdataframe(index)
        elif isinstance(index, tuple):
            return self._get_multiple_selection(index)
        raise TypeError(
            f"Must pass either a [`str`, `list`, `DataFrame`, `tuple`] - got {type(index)} instead."
        )

    def _get_single_column(self, index: str):
        if not index in self.columns:
            raise ValueError(f"`{index}` not found in columns: {self.columns}")
        return DataFrame({index: self._data[index]})

    def _get_multiple_columns(self, indexes: List[str]):
        data = {}
        for column in indexes:
            if column not in self.columns:
                raise ValueError(f"`{column}` not found in columns: {self.columns}")
            else:
                data[column] = self._data[column]
        return DataFrame(data)

    def _get_subdataframe(self, index_df):
        if not index_df.shape[1] == 1:
            raise ValueError(
                f"Indexing is supported only for single column DataFrames- got {len(self.columns)}"
            )
        array = next(iter(index_df._data.values()))
        if array.dtype.kind != "b":
            raise TypeError(
                f"Indexing is supported only for DataFrames of dtype `bool`- got {DTYPES[array.dtype.kind]}"
            )
        return DataFrame(
            {col_name: value[array] for col_name, value in self._data.items()}
        )

    def _get_multiple_selection(self, indexes):
        if len(indexes) != 2:
            raise ValueError(
                f"Must pass in a tuple of length 2- got {len(indexes)} instead"
            )
        row_selection, col_selection = indexes
        if isinstance(row_selection, int):
            row_selection = [row_selection]
        elif isinstance(row_selection, DataFrame):
            if row_selection.shape[1] != 1:
                raise ValueError(
                    f"Row selection DataFrame should be single column- got {row_selection.shape[1]} instead"
                )
            row_selection = next(iter(row_selection._data.values()))
            if row_selection.dtype.kind != "b":
                raise TypeError(
                    f"Row selection DataFrame should be of type `bool`- got {DTYPES[row_selection.dtype.kind]} instead"
                )
        elif not isinstance(row_selection, (list, slice)):
            raise TypeError(
                f"Row selection must be of one of the types [`list`, `slice`, `DataFrame`, `int`] - got {type(row_selection)} instead."
            )
        if isinstance(col_selection, int):
            col_selection = self.columns[col_selection]
        elif isinstance(col_selection, str):
            col_selection = [col_selection]
        elif isinstance(col_selection, list):
            new_col_selection = []
            for col in col_selection:
                if isinstance(col, int):
                    new_col_selection.append(self.columns[col])
                elif isinstance(col, str):
                    new_col_selection.append(col)
                else:
                    raise TypeError(
                        "Column selection list values should be of type [`str`, `int`]"
                    )
                col_selection = new_col_selection
        elif isinstance(col_selection, slice):
            start = col_selection.start
            stop = col_selection.stop
            step = col_selection.step  # could be None or str or int
            if isinstance(start, str):
                start = self.columns.index(start)
            if isinstance(stop, str):
                start = self.columns.index(stop) + 1
            col_selection = self.columns[start:stop:step]

        else:
            raise TypeError(
                "Column selection must be of type [`int`, `slice`, `list`, `str`]"
            )
        data = {}
        for col in col_selection:
            data[col] = self._data[col][row_selection]

        return DataFrame(data)

    def _ipython_key_completions_(self):
        """
            Tab completion for column indexing in Jupyter
        """
        return self.columns


if __name__ == "__main__":
    a = np.array(["a", "b", "c"])
    b = np.array(["c", "d", None])
    c = np.random.rand(3)
    d = np.array([True, False, True])
    e = np.array([1, 2, 3])
    df = DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
