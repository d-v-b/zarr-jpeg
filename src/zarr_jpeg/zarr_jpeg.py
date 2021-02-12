from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray, ensure_contiguous_ndarray, ndarray_copy
from numcodecs.registry import register_codec
from imagecodecs import jpeg_encode, jpeg_decode


class jpeg(Codec):
    """Codec providing jpeg compression via imagecodecs.

    Parameters
    ----------
    quality : int
        Compression level.
    """

    codec_id = "jpeg"

    def __init__(self, quality=100, axis_reduction='collapse'):
        self.quality = quality
        self.axis_reduction = axis_reduction
        assert (
            self.quality > 0 and self.quality <= 100 and isinstance(self.quality, int)
        )
        super().__init__()

    def encode(self, buf):
        bufa = ensure_ndarray(buf)
        assert bufa.ndim >= 2
        axis_reduction = self.axis_reduction
        if isinstance(axis_reduction, str) and axis_reduction == 'collapse':
            if bufa.ndim >= 3:
                # all but the last dimension are collapsed into first dimension, followed by last dimension
                axis_reduction = [[dim for dim in range(bufa.ndim - 1)]]
                axis_reduction.append([bufa.ndim - 1])
            else:
                # keep each dimension as is.
                axis_reduction = None

        if axis_reduction is None:
            tiled = bufa
        else:
            # Check that each dimension is mentioned exactly once, in order.
            assert [dim for axis in axis_reduction for dim in axis] == [dim for dim in range(buf.ndim)]
            # Reshape using the axis_reduction
            tiled = bufa.reshape([np.prod([bufa.shape[dim] for dim in axis], dtype='uint') for axis in axis_reduction])

        return jpeg_encode(tiled, level=self.quality)

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf)

        if out is not None:
            out = ensure_contiguous_ndarray(out)

        tiled = jpeg_decode(buf)
        return ndarray_copy(tiled, out)


register_codec(jpeg)
