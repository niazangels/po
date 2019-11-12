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

