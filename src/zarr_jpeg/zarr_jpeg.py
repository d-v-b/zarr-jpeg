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
    
    def __init__(self, quality=100):
        self.quality = quality
        assert self.quality > 0 and self.quality <= 100 and isinstance(self.quality, int)
        super().__init__()
    
    def encode(self, buf):
        bufa = ensure_ndarray(buf)        
        assert (bufa.ndim == 2 or bufa.ndim == 3)
        if bufa.ndim == 3:
            tiled = bufa.reshape(bufa.shape[0] * bufa.shape[1], -1)
        else: 
            tiled = bufa
        return jpeg_encode(tiled, level=self.quality)
    
    def decode(self, buf, out=None):                  
        buf = ensure_contiguous_ndarray(buf)
        
        if out is not None:           
            out = ensure_contiguous_ndarray(out)
    
        tiled = jpeg_decode(buf)                
        return ndarray_copy(tiled, out)

register_codec(jpeg)