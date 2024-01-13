import re

import numpy
import numpy as np
from pathlib import Path

import skimage.measure.block
import os

import math
from numpy import ndarray
from tqdm import tqdm
import png


def mapValue(value, fromRange, toRange):
    # Figure out how 'wide' each range is
    fromSpan = fromRange[0] - fromRange[1]
    toSpan = toRange[0] - toRange[1]

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - fromRange[0]) / float(fromSpan)

    # Convert the 0-1 range into a value in the right range.
    return toRange[0] + (valueScaled * toSpan)


def coordsToIndex(x, y, nCols) -> int:
    return y * nCols + x


def downscale(x, y, factor) -> tuple:
    return (x*factor,y*factor)


class ASCFile:
    nCols: int
    nRows: int
    minX: float
    minY: float
    rowcount: int
    cellsize: float
    minAltitude: float
    maxAltitude: float
    _data: ndarray
    _filepath: Path

    def __init__(self, path, metadata_only=True) -> None:
        rowcount = 0
        self._filepath = path
        with open(path, 'r', encoding="utf8") as file:
            self.name = os.path.basename(file.name)
            self.maxAltitude = 0
            self.minAltitude = 9999
            datalines = []
            try:
                for line in file:
                    rowcount += 1
                    if rowcount == 1:
                        self.nCols = int(re.search("[0-9.]+", line).group())
                    elif rowcount == 2:
                        self.nRows = int(re.search("[0-9.]+", line).group())
                    elif rowcount == 3:
                        self.minX = float(re.search("[0-9.]+", line).group())
                    elif rowcount == 4:
                        self.minY = float(re.search("[0-9.]+", line).group())
                    elif rowcount == 5:
                        self.cellSize = float(re.search("[0-9.]+", line).group())
                    elif rowcount == 6:
                        pass
                    elif rowcount >= 7:
                        self.maxAltitude = float(np.max([self.maxAltitude]+[float(digit) for digit in line.split()]))
                        self.minAltitude = float(np.min([self.minAltitude]+[float(digit) for digit in line.split()]))
            except:
                print("Cannot parse "+ self.name)
                raise
            self.maxX = self.minX + (self.nCols * self.cellSize)
            self.maxY = self.minY + (self.nRows * self.cellSize)
            self.size = (self.maxX-self.minX, self.maxY-self.minY)

    def index2coord(self, index, scale=1) -> tuple:
        return index % math.ceil(self.nCols/scale), index // math.ceil(self.nCols/scale)

    @staticmethod
    def map_value_to_int8_color_value(map_value) -> int:
        _maxColorValue = 0xff
        _maxMapValue = 4100
        mapvalue_relative_to_max = map_value / _maxMapValue
        mapped_value = _maxColorValue * mapvalue_relative_to_max
        return round(mapped_value)

    @staticmethod
    def map_value_to_fmode_color_value(map_value) -> float:
        _maxColorValue = 0x7FFFFFFF
        _maxMapValue = 4100
        mapvalue_relative_to_max = map_value / _maxMapValue
        mapped_value = float(_maxColorValue * mapvalue_relative_to_max)
        return mapped_value

    def get_altitude(self, x, y) -> float:
        return self.data()[x][y]

    def data(self) -> ndarray:
        try:
            return self._data
        except AttributeError:
            with open(self._filepath, 'r', encoding="utf8") as f:
                datalines = []
                rowcount = 0
                for line in f:
                    rowcount += 1
                    if rowcount < 7:
                        pass
                    else:
                        datalines.append([float(digit) for digit in line.split()])
                self._data = np.concatenate(datalines)
                self._data = np.reshape(self._data, (self.nCols, self.nRows))
                self._data = np.rot90(self._data, -1)
                return self._data

    def coord2index(self, x, y) -> int:
        return x * self.nCols + y


def process_files(inpaths: list, outpath: Path, downscalefactor=1, relativeAltitude=False) -> None:
    ascfiles = []
    for inpath in tqdm(inpaths, desc="Reading ASC files"):
        ascfiles.append(ASCFile(inpath))

    assert all([asc.cellSize == ascfiles[0].cellSize for asc in ascfiles])  # Check that cellsize is uniform across all files
    cellSize = ascfiles[0].cellSize
    assert all([asc.nCols == ascfiles[0].nCols for asc in ascfiles])  # Check that nCols is uniform across all files
    nCols = ascfiles[0].nCols
    assert all([asc.nRows == ascfiles[0].nRows for asc in ascfiles])  # Check that nRows is uniform across all files
    nRows = ascfiles[0].nRows

    assert nCols % downscalefactor == 0  # TODO: remove, good luck lmao

    imageminX = min([f.minX for f in ascfiles])
    imagemaxX = max([f.maxX for f in ascfiles])
    imageminY = min([f.minY for f in ascfiles])
    imagemaxY = max([f.maxY for f in ascfiles])
    imageminZ = min([f.minAltitude for f in ascfiles])
    imagemaxZ = max([f.maxAltitude for f in ascfiles])

    size_x = int(math.ceil((imagemaxX - imageminX) / cellSize / downscalefactor))
    size_y = int(math.ceil((imagemaxY - imageminY) / cellSize / downscalefactor))

    thebigarray = np.zeros((size_x*2, size_y), dtype=np.int32) # *2 because 2 channels. Default value of 0 = transparent unless worked on later

    for ascFile in tqdm(ascfiles, desc="Building image"):
        downsampled_image = skimage.measure.block.block_reduce(ascFile.data(), block_size=downscalefactor, func=numpy.mean)
        for index in range(0, downsampled_image.size):
            x, y = ascFile.index2coord(index, downscalefactor)
            imagex = (x + ((ascFile.minX - imageminX) / cellSize) / downscalefactor )
            imagey = (y + ((ascFile.minY - imageminY) / cellSize) / downscalefactor )
            try:
                thebigarray[int(imagex*2), int(imagey)] = int(mapValue(downsampled_image[x,y], (0, 4500), (0x0, 0x7fff))) # Color
                thebigarray[int(imagex*2+1), int(imagey)] = 0xffff #Transparency
            except Exception:
                print("!!!")
                print(ascFile.data()[x*downscalefactor,y*downscalefactor])
                print(downsampled_image[x,y])
                print(index)
                print("!!!")
                raise
    thebigarray = np.rot90(thebigarray)
    img = png.from_array(thebigarray.tolist(), mode="LA;16")
    img.save(outpath)



if __name__ == "__main__":
    chartreuse = [
        Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0925_6475_MNT_LAMB93_IGN69.asc"),
        Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0925_6500_MNT_LAMB93_IGN69.asc"),
        Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0900_6475_MNT_LAMB93_IGN69.asc"),
        #Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0900_6500_MNT_LAMB93_IGN69.asc"),
    ]

    oneimg = [Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0925_6475_MNT_LAMB93_IGN69.asc")]
    isere = [Path("S:/Curiosités/IGN/BD ALTI - 38") / Path(filename) for filename in os.listdir(Path("S:/Curiosités/IGN/BD ALTI - 38"))]
    france = [Path("S:/Curiosités/IGN/BD ALTI - France") / Path(filename) for filename in os.listdir(Path("S:/Curiosités/IGN/BD ALTI - France"))]

    process_files(chartreuse, Path("S:/Curiosités/IGN/chartreuse_transp_ds2.png"), downscalefactor=2)

    print("OK!")
