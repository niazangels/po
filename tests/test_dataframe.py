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


class TestDataFrameSelection:
    def test_get_single_column(self):
        with pytest.raises(ValueError):
            df["invalid_column"]
        assert_df_equals(df["a"], po.DataFrame({"a": a}))

    def test_get_multiple_column(self):
        with pytest.raises(ValueError):
            df[["a", "car"]]
        assert_df_equals(df[["a", "b", "c", "d", "e"]], df)

    def test_get_sub_dataframe(self):
        with pytest.raises(ValueError):
            df[df]
        with pytest.raises(TypeError):
            df_index = po.DataFrame({"a": a})
            df[df_index]

        index_df = po.DataFrame({"selection": np.ones(len(a)).astype("bool")})
        assert_df_equals(df[index_df], df)

    def test_single_element(self):
        assert_df_equals(df[1, 3], po.DataFrame({"d": np.array([False])}))

    def test_multiple_row(self):
        # TODO: raises errors
        filt = po.DataFrame({"f": np.array([False, True, True])})

        expected_df = po.DataFrame({"a": np.array(["b"])})
        assert_df_equals(df[1, 0], expected_df)

        # Row selection variants

        with pytest.raises(TypeError):
            _ = df["0", 0]

        expected_df = po.DataFrame({"a": np.array(["a"])})
        assert_df_equals(df[0, 0], expected_df)

        expected_df = po.DataFrame({"a": np.array(["b", "c"])})
        assert_df_equals(df[[1, 2], 0], expected_df)
        assert_df_equals(df[1:3, 0], expected_df)
        assert_df_equals(df[filt, 0], expected_df)

        # Col selection variants
        with pytest.raises(TypeError):
            _ = df[0, 0.2]

        expected_df = po.DataFrame({"e": np.array([1, 3])})
        assert_df_equals(df[[0, 2], "e"], expected_df)
        assert_df_equals(df[[0, 2], 4], expected_df)

        expected_df = po.DataFrame(
            {"d": np.array([True, True]).astype(bool), "e": np.array([1, 3])}
        )
        assert_df_equals(df[[0, 2], ["d", "e"]], expected_df)
        assert_df_equals(df[[0, 2], [3, 4]], expected_df)
        assert_df_equals(df[[0, 2], -2:], expected_df)

    def test_key_completion(self):
        assert df._ipython_key_completions_() == df.columns
