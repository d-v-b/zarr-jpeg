# zarr-jpeg

### About
Enable JPEG encoding of zarr / n5 chunks. Only tested with 2D or 3D `uint8` numeric data. This encoding is lossy. Because jpeg encoding only works on 2D arrays, this implementation "planarizes" 3D data before encoding by concatenating the first and second axes of the data using `np.reshape`, which does not incur a data copy. I.e., if one wishes to encode an array with shape = (4,5,6), this array is reshaped to (20,6) before compression; The inverse procedure occurs when decoding data. Because of this planarization procedure, you will get the best compression ratio when the spatial correlation of your data is highest in the space defined by the last two axes. To put it another way, for best compression results ensure that your intensity values vary the most along the first axis of your data.

### Usage

Stand-alone:
```python
from zarr_jpeg import jpeg
import numpy as np
data = np.random.randint((100,255), (100,100,100), dtype='uint8')
codec = jpeg(quality=100)
encoded = codec.encode(data)
# on decoding, the original shape is unknown, so we have to reshape ourselves
decoded = codec.decode(encoded).reshape(data.shape)
```
With zarr:
```python
from zarr_jpeg import jpeg
import zarr
array = zarr.open_array('foo/bar.zarr', path='path/to/array', compressor=jpeg(quality=50), shape=(100,100,100), dtype='uint8')
```

### References
This repo is inspired by the [neuroglancer "precomputed" format](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md), which uses jpeg encoding to compress chunks of imaging data.