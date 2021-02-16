import numpy as np
import pytest
from imagecodecs import jpeg_encode
from zarr_jpeg import jpeg
from zarr_jpeg.zarr_jpeg import validate_axis_reduction
from numcodecs.tests.common import check_config
from itertools import product

codecs = [
    jpeg(input_shape=(8,8)),
    jpeg(input_shape=(8,8,8), quality=1),
    jpeg(input_shape=(8,8,8), quality=1),
    jpeg(input_shape=(8,9,10,11), quality=100),
]


# only uint8 data
# 2D shapes and larger
# mix of orders: C, F
arrays = [
    np.zeros((8,8), dtype='uint8'),
    np.random.randint(0, 255, size=8 * 8 * 8, dtype="u1").reshape(8,8,8, order="F"),
    np.random.randint(0, 255, size=8 * 8 * 8, dtype="u1").reshape(8,8,8, order="C"),
    np.random.randint(0, 255, size=8 * 9 * 10 * 11, dtype="u1").reshape(8, 9, 10, 11, order="C"),
]


def test_config():
    codec = jpeg(input_shape=(10,10), quality=50)
    check_config(codec)


def test_eq():
    input_shape = (8,8)
    assert jpeg(input_shape) == jpeg(input_shape)
    assert not jpeg(input_shape) != jpeg(input_shape)
    assert jpeg(input_shape, quality=1) == jpeg(input_shape, quality=1)
    assert jpeg(input_shape, quality=1) != jpeg(input_shape, quality=90)
    assert jpeg(input_shape) != "foo"
    assert "foo" != jpeg(input_shape)
    assert not jpeg(input_shape) == "foo"


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


def test_default_axis_reduction():
    for array, codec in zip(arrays, codecs):
        print(array.shape)
        print(codec.axis_reduction)
        new_shape = [np.prod([array.shape[dim] for dim in axis], dtype='int') for axis in codec.axis_reduction]
        print(new_shape)
        assert codec.encode(array) == jpeg_encode(
            array.reshape(new_shape), level=codec.quality)


def test_validate_axis_reduction():
    input_shape=(1,)
    axis_reduction = None
    with pytest.raises(ValueError):
        validate_axis_reduction(input_shape, axis_reduction)

    input_shape=(1,1)
    axis_reduction = None
    with pytest.raises(ValueError):
        validate_axis_reduction(input_shape, axis_reduction)

    input_shape=(10,1,1)
    axis_reduction = None
    with pytest.raises(ValueError):
        validate_axis_reduction(input_shape, axis_reduction)

    input_shape=(10,10)
    axis_reduction = None
    assert validate_axis_reduction(input_shape, axis_reduction) == ((0,), (1,), ())

    input_shape=(10,10,10)
    axis_reduction = None
    assert validate_axis_reduction(input_shape, axis_reduction) == ((0,1), (2,), ())

    input_shape=(10,10,10,1)
    axis_reduction = None
    assert validate_axis_reduction(input_shape, axis_reduction) == ((0,1,3), (2,), ())

    input_shape=(10,10,1,1)
    axis_reduction = None
    assert validate_axis_reduction(input_shape, axis_reduction) == ((0,2,3), (1,), ())
