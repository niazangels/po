from numpy.testing import assert_allclose, assert_array_equal


def assert_df_equals(df1, df2):
    assert df1.columns == df2.columns

    for v1, v2 in zip(df1._data.values(), df2._data.values()):
        assert v1.dtype.kind == v2.dtype.kind
        if v1.dtype.kind == "f":
            assert_allclose(v1, v2)
        else:
            assert_array_equal(v1, v2)
