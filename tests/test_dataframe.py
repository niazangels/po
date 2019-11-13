import po
import numpy as np
import pytest

a = np.array(["a", "b", "c"])
b = np.array(["c", "d", None])
c = np.random.rand(3)
d = np.array([True, False, True])
e = np.array([1, 2, 3])


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

