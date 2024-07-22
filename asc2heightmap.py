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
from typing import Callable
import itertools


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
    return x*factor, y*factor


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

    def __init__(self, path) -> None:
        rowcount = 0
        self._filepath = path
        with open(path, 'r', encoding="utf8") as file:
            self.name = os.path.basename(file.name)
            self._maxAltitude = None
            self._minAltitude = None
            try:
                for line in itertools.islice(file, 6):  # Elems 0 to 5 inclusive
                    rowcount += 1
                    match rowcount:
                        case 1:
                            self.nCols = int(re.search("[0-9.]+", line).group())
                        case 2:
                            self.nRows = int(re.search("[0-9.]+", line).group())
                        case 3:
                            self.minX = float(re.search("[0-9.]+", line).group())
                        case 4:
                            self.minY = float(re.search("[0-9.]+", line).group())
                        case 5:
                            self.cellSize = float(re.search("[0-9.]+", line).group())
            except Exception:
                print("Cannot parse " + self.name)
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

    @property
    def minAltitude(self) -> float:
        try:
            return self._minAltitude
        except AttributeError:
            with open(self._filepath, 'r', encoding="utf8") as file:
                for line in itertools.islice(file, 7, None):
                    m = float(np.min([self.minAltitude]+[float(digit) for digit in line.split()]))
            self._minAltitude = m
            return self._minAltitude

    @property
    def maxAltitude(self) -> float:  # Lazyload
        try:
            return self._minAltitude
        except AttributeError:
            with open(self._filepath, 'r', encoding="utf8") as file:
                for line in itertools.islice(file, 7, None):
                    m = float(np.min([self.minAltitude]+[float(digit) for digit in line.split()]))
            self._minAltitude = m
            return self._minAltitude

    @property
    def data(self) -> ndarray:
        try:
            return self._data
        except AttributeError:
            self._data = np.loadtxt(self._filepath, dtype=float, skiprows=7, encoding='utf-8')
            return self._data


    def coord2index(self, x, y) -> int:
        return x * self.nCols + y


def load_ascfiles(inpaths: list):
    ascfiles = []
    for inpath in tqdm(inpaths, desc="Reading ASC files"):
        ascfiles.append(ASCFile(inpath))
    return ascfiles


def process_files(inpaths: list, outpath: Path, downscalefactor: int = 1, transparency_on_empty: bool = False, modifier: Callable = None, relativeAltitude=False) -> None:
    if modifier is None:
        def modifier(value): return value

    ascfiles = load_ascfiles(inpaths)

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
    # imageminZ = min([f.minAltitude for f in ascfiles])
    # imagemaxZ = max([f.maxAltitude for f in ascfiles])

    size_x = int(math.ceil((imagemaxX - imageminX) / cellSize / downscalefactor))
    size_y = int(math.ceil((imagemaxY - imageminY) / cellSize / downscalefactor))

    if transparency_on_empty:
        thebigarray = np.zeros((size_x*2, size_y), dtype=np.int32)  # *2 because 2 channels. Default value of 0 = transparent unless worked on later
    else:
        thebigarray = np.zeros((size_x, size_y), dtype=np.int32)

    for ascFile in tqdm(ascfiles, desc="Building image"):
        downsampled_image = skimage.measure.block.block_reduce(ascFile.data, block_size=downscalefactor, func=numpy.mean)
        for index in range(0, downsampled_image.size):
            x, y = ascFile.index2coord(index, downscalefactor)
            imagex = (x + ((ascFile.minX - imageminX) / cellSize) / downscalefactor)
            imagey = (y + ((ascFile.minY - imageminY) / cellSize) / downscalefactor)
            try:
                if transparency_on_empty:
                    thebigarray[int(imagex*2),   int(imagey)] = int(mapValue(modifier(downsampled_image[x, y]), (modifier(0), modifier(4500)), (0x0, 0x7fff)))  # Color
                    thebigarray[int(imagex*2+1), int(imagey)] = 0xffff  # Transparency
                else:
                    thebigarray[int(imagex),     int(imagey)] = int(mapValue(modifier(downsampled_image[x, y]), (modifier(0), modifier(4500)), (0x0, 0x7fff)))
            except Exception:
                print("!!!")
                print(ascFile.data[x*downscalefactor, y*downscalefactor])
                print(downsampled_image[x, y])
                print(index)
                print("!!!")
                raise
    thebigarray = np.rot90(thebigarray)
    if transparency_on_empty:
        img = png.from_array(thebigarray.tolist(), mode="LA;16")
    else:
        img = png.from_array(thebigarray.tolist(), mode="L;16")
    img.save(outpath)


def benchmark():
    import cProfile
    import pstats
    statsFile = Path("C:/Users/Antonin/AppData/Local/Temp/profilestats.txt")

    def benchmark_worker():
        isere = [Path("S:/Curiosités/IGN/BD ALTI - 38") / Path(filename) for filename in
                 os.listdir(Path("S:/Curiosités/IGN/BD ALTI - 38"))]
        process_files(isere, Path("C:/Users/Antonin/AppData/Local/Temp/profile.png"), downscalefactor=4,
                      transparency_on_empty=True)

    cProfile.run("benchmark_worker()", str(statsFile))
    p = pstats.Stats(str(statsFile))
    return p


def main():
    chartreuse = [
        Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0925_6475_MNT_LAMB93_IGN69.asc"),
        Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0925_6500_MNT_LAMB93_IGN69.asc"),
        Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0900_6475_MNT_LAMB93_IGN69.asc"),
        Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0900_6500_MNT_LAMB93_IGN69.asc"),
    ]

    oneimg = [Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0925_6475_MNT_LAMB93_IGN69.asc")]
    isere = [Path("S:/Curiosités/IGN/BD ALTI - 38") / Path(filename) for filename in os.listdir(Path("S:/Curiosités/IGN/BD ALTI - 38"))]
    france = [Path("S:/Curiosités/IGN/BD ALTI - France") / Path(filename) for filename in os.listdir(Path("S:/Curiosités/IGN/BD ALTI - France"))]

    dsf: int = 4
    transp: bool = True
    name: str = "isere"
    filename: Path = Path("S:/Curiosités/IGN/") / f"{name}_ds{dsf}_transp{transp}.png"

    process_files(isere, filename, dsf, transp)

    print("OK! " + str(filename))

if __name__ == "__main__":
    main()
