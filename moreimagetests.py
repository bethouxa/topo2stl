import numpy as np
import tifffile
from pathlib import Path

savePath = Path("C:/Users/Antonin/Desktop/test.tiff")

#image = np.random.rand(500, 500, 3).astype(np.float32)

npa = np.array([[1.1,2.2,3.3],[4.4,5.5,6.6],[7.7,8.8,9.9]],ndmin=2,dtype=np.float32)

tifffile.imsave(savePath, image)