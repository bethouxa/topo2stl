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
from datetime import datetime


def mapValue(value, fromRange: tuple, toRange: tuple):
    # Figure out how 'wide' each range is
    fromSpan = fromRange[0] - fromRange[1]
    toSpan = toRange[0] - toRange[1]

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - fromRange[0]) / float(fromSpan)

    # Convert the 0-1 range into a value in the right range.
    return toRange[0] + (valueScaled * toSpan)


def coordsToIndex(x: int, y: int, nCols: int) -> int:
    return y * nCols + x


def downscale(x: int, y: int, factor: int) -> tuple:
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

    def __init__(self, path: Path) -> None:
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

    def index2coord(self, index: int, scale: int = 1) -> tuple:
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

    image_min_X = min([f.minX for f in ascfiles])
    image_max_X = max([f.maxX for f in ascfiles])
    image_min_Y = min([f.minY for f in ascfiles])
    image_max_Y = max([f.maxY for f in ascfiles])
    #imageminZ = min([f.minZ for f in ascfiles])
    #imagemaxZ = max([f.maxZ for f in ascfiles])

    image_size_X = int(math.ceil((image_max_X - image_min_X) / cellSize / downscalefactor))
    image_size_Y = int(math.ceil((image_max_Y - image_min_Y) / cellSize / downscalefactor))

    if transparency_on_empty:
        thebigarray = np.zeros((image_size_X*2, image_size_Y), dtype=np.int32)
        # *2 for color channel +alpha channel. Default value of 0 = transparent unless worked on later
    else:
        thebigarray = np.zeros((image_size_X, image_size_Y), dtype=np.int32)
        # No alpha channel

    for chunk in tqdm(ascfiles, desc="Building image"):
        # Downsample => for example, bring 1000x1000 image to 500x500 image
        downsampled_chunk = skimage.measure.block.block_reduce(chunk.data, block_size=downscalefactor, func=numpy.mean)
        for index in range(0, downsampled_chunk.size):
            chunk_x, chunk_y = chunk.index2coord(index, downscalefactor)
            # Compute the positions of the "current pixel" in the final image
            image_x = (chunk_x + ((chunk.minX - image_min_X) / cellSize) / downscalefactor)
            image_y = (chunk_y + ((chunk.minY - image_min_Y) / cellSize) / downscalefactor)
            try:
                if transparency_on_empty:
                    thebigarray[int(image_x*2),   int(image_y)] = int(mapValue(modifier(downsampled_chunk[chunk_x, chunk_y]), (modifier(0), modifier(4500)), (0x0, 0x7fff)))  # Color
                    thebigarray[int(image_x*2+1), int(image_y)] = 0xffff  # Transparency
                else:
                    thebigarray[int(image_x),     int(image_y)] = int(mapValue(modifier(downsampled_chunk[chunk_x, chunk_y]), (modifier(0), modifier(4500)), (0x0, 0x7fff)))
            except Exception:
                print("!!!")
                print("ascfile data:      " + str(chunk.data[chunk_x*downscalefactor, chunk_y*downscalefactor]))
                print("downsampled image: " + str(downsampled_chunk[chunk_x, chunk_y]))
                print("index:             " + str(index))
                print("!!!")
                raise
    thebigarray = np.rot90(thebigarray)
    if transparency_on_empty:
        img = png.from_array(thebigarray.tolist(), mode="LA;16")
    else:
        img = png.from_array(thebigarray.tolist(), mode="L;16")
    img.save(outpath)


def benchmark_worker():
    isere = [Path("S:/Curiosités/IGN/BD ALTI - 38") / Path(filename) for filename in os.listdir(Path("S:/Curiosités/IGN/BD ALTI - 38"))]
    process_files(isere, Path("C:/Users/betho/AppData/Local/Temp/profile.png"), downscalefactor=4, transparency_on_empty=True)


def benchmark():
    import cProfile
    import pstats
    statsFile = Path("C:/Users/betho/AppData/Local/Temp/profilestats.txt")
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

    dsf: int = 2
    transp: bool = False
    name: str = "isere2"
    date = datetime.today().strftime('%Y-%m-%d')
    filename: Path = Path("S:/Curiosités/IGN/") / f"{name}_ds{dsf}_transp{transp}_{date}.png"

    process_files(isere, filename, dsf, transp)

    print("OK! " + str(filename))


if __name__ == "__main__":
    main()
