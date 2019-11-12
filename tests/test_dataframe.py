import po
import numpy as np
import pytest


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

