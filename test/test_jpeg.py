import itertools


import numpy as np
import pytest


from zarr_jpeg import jpeg
from numcodecs.tests.common import check_encode_decode, check_config, check_repr

from itertools import product

codecs = [
    jpeg(),
    jpeg(quality=1),
    jpeg(quality=50),
    jpeg(quality=100),
]


# only uint8 data
# only 2D and 3D shapes
# mix of orders: C, F
arrays = [
    np.arange(100, dtype="u1").reshape((10, 10)),
    np.random.randint(0, 255, size=10 * 11 * 12, dtype="u1").reshape(
        10, 11, 12, order="F"
    ),
    np.random.randint(0, 255, size=10 * 11 * 12, dtype="u1").reshape(
        10, 11, 12, order="C"
    ),
]


def test_config():
    codec = jpeg(quality=50)
    check_config(codec)


def test_eq():
    assert jpeg() == jpeg()
    assert not jpeg() != jpeg()
    assert jpeg(1) == jpeg(1)
    assert jpeg(1) != jpeg(90)
    assert jpeg() != "foo"
    assert "foo" != jpeg()
    assert not jpeg() == "foo"


def test_err_encode_list():
    data = ["foo", "bar", "baz"]
    for codec in codecs:
        with pytest.raises(TypeError):
            codec.encode(data)


def test_err_out_too_small():
    arr = np.arange(120, dtype="u1").reshape(15, 8)
    out = np.empty_like(arr)[:-1]
    for codec in codecs:
        with pytest.raises(ValueError):
            codec.decode(codec.encode(arr), out)


def test_planarization():
    for array, codec in product(arrays, codecs):
        if array.ndim > 2:
            assert codec.encode(array) == codec.encode(
                array.reshape(array.shape[0] * array.shape[1], -1)
            )
