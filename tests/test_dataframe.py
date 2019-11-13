import po
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tests import assert_df_equals

a = np.array(["a", "b", "c"])
b = np.array(["c", "d", None])
c = np.random.rand(3)
d = np.array([True, False, True])
e = np.array([1, 2, 3])
df = po.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})


class TestDataFrameCreation:
    def test_input_types(self):
        with pytest.raises(TypeError):
            po.DataFrame(1)
        with pytest.raises(TypeError):
            po.DataFrame({1: [1, 2, 3]})
        with pytest.raises(TypeError):
            po.DataFrame({"hello": np.ndarray([[1]])})

    def test_input_lengths(self):
        with pytest.raises(ValueError):
            po.DataFrame({"a": np.array([1, 2, 3]), "b": np.array([1, 2])})

        po.DataFrame({"a": np.array([1, 2, 3]), "b": np.array([1, 2, 3])})

    def test_unicode_to_object(self):
        df = po.DataFrame({"a": a})
        assert df._data["a"].dtype == "object"

    def test_dunder_len(self):
        df = po.DataFrame({"a": a})
        assert len(df) == len(a)

    def test_columns_getter(self):
        assert df.columns == list("abcde")

    def test_columns_setter(self):
        with pytest.raises(TypeError):
            df.columns = "invalid"
        with pytest.raises(ValueError):
            df.columns = ["1", "2", "3"]
        with pytest.raises(TypeError):
            df.columns = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            df.columns = ["1", "3", "3", "4", "5"]
        df.columns = ["1", "2", "3", "4", "5"]
        df.columns = ["a", "b", "c", "d", "e"]

    def test_shape(self):
        assert df.shape == (len(a), len(df._data))

    def test_values(self):
        assert_array_equal(df.values, np.array([a, b, c, d, e]))

    def test_dtypes(self):
        cols = np.array(["a", "b", "c", "d", "e"], dtype="O")
        dtypes = np.array(["string", "string", "float", "bool", "int"], dtype="O")

        df_result = df.dtypes
        df_answer = po.DataFrame({"column_name": cols, "dtype": dtypes})
        assert_df_equals(df_result, df_answer)

