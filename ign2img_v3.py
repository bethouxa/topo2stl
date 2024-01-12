import re

import numpy
import numpy as np
from pathlib import Path

import skimage.measure.block
import tifffile
import os

import math
from numpy import ndarray
from tqdm import tqdm
import png
from concurrent.futures import ThreadPoolExecutor, wait


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

    def get_color_value(self, x, y) -> int:
        return self.map_value_to_8b_color_value(self.get_altitude(x, y))

    def coord2index(self, x, y) -> int:
        return x * self.nCols + y

    def downscale(self, x, y, factor) -> tuple:
        return (x*factor,y*factor)


class ASCPuzzle:
    maxAlt: int
    minAlt: int
    files: list

    def __init__(self, ASCFiles):
        maxAlt = max([f.maxAltitude for f in ASCFiles])
        minAlt = min([f.minAltitude for f in ASCFiles])
        minX = min([f.minX for f in ASCFiles])
        maxX = max([f.maxX for f in ASCFiles])
        minY = min([f.minY for f in ASCFiles])
        maxY = max([f.maxY for f in ASCFiles])
        for f in ASCFiles:
            pass

        def getContainingASCFile(self, x, y):
            if x < self.minX or x > self.maxX or y < minY or y > maxY:
                raise ValueError
            for ascfile in self.ASCFiles:
                if ascfile.minX < x < ascfile.minX + ascfile.nCols * ascfile.cellsize and ascfile.minY < y < ascfile.minY + ascfile.nCols * ascfile.cellsize:
                    return ascfile


def one_image(inpath, outpath) -> None:
    asc = ASCFile(inpath)
    a = np.array(asc.data, dtype=np.int32, ndmin=2)
    tifffile.imwrite(outpath, a, shape=a.shape, dtype=np.int32)
#    img = Image.new('RGB', (asc.sizeX, asc.sizeY))
#    for y in range(0, asc.sizeY):
#        for x in range(0, asc.sizeX):
#            color = asc.get_color_value(x, y)
#            img.putpixel((x, y), (color, color, color))
#    img.save(Path(outpath))


def multiple_images(inpaths, outpath: Path) -> None:
    ascfiles = []
    for inpath in tqdm(inpaths, desc='Reading ASC files'):
        ascfiles.append(ASCFile(inpath))

    assert all([asc.cellSize == ascfiles[0].cellSize for asc in ascfiles])  # Check that cellsize is uniform across all files
    cellSize = ascfiles[0].cellSize
    assert all([asc.nCols == ascfiles[0].nCols for asc in ascfiles])  # Check that nCols is uniform across all files
    nCols = ascfiles[0].nCols
    assert all([asc.nRows == ascfiles[0].nRows for asc in ascfiles])  # Check that nRows is uniform across all files
    nRows = ascfiles[0].nRows

    minX = min([f.minX for f in ascfiles])
    maxX = max([f.maxX for f in ascfiles])
    minY = min([f.minY for f in ascfiles])
    maxY = max([f.maxY for f in ascfiles])
    offsetX = minX
    offsetY = minY
    size_x = int((maxX - minX) / cellSize)  # Assuming cellSize is the same for all files
    size_y = int((maxY - minY) / cellSize)  # Assuming cellSize is the same for all files

    thebigarray = np.zeros((size_x, size_y), dtype=np.int32)

    for file in tqdm(ascfiles, desc="Building image"):

        for index in range(0, nCols*nRows):
            x, y = file.index2coord(index)
            localx = (x + ((file.minX - offsetX) / cellSize))
            localy = (y + ((file.minY - offsetY) / cellSize))
            try:
                thebigarray[int(localx), int(localy)] = int(mapValue(file.data()[x,y], (0, 4500), (0x0, 0x7fff)))
            except Exception:
                print("!!!")
                print(file.data()[x,y])
    #tifffile.imwrite(outpath, thebigarray, shape=thebigarray.shape, dtype=np.int32)
    thebigarray = np.rot90(thebigarray)
    thebigarray = np.rot90(thebigarray)
    img = png.from_array(thebigarray.tolist(), mode='L;16')
    img.save(outpath)


def multiple_images_scaled(inpaths: list, outpath: Path, downscalefactor=1, relativeAltitude=False) -> None:
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

    thebigarray = np.zeros((size_x, size_y), dtype=np.int32)

    for ascFile in tqdm(ascfiles, desc="Building image"):

        # sampleCount = nCols * nRows // downscalefactor**2
        downsampled_image = skimage.measure.block.block_reduce(ascFile.data(), block_size=downscalefactor, func=numpy.mean)

        for index in range(0, downsampled_image.size):

            # Get final image position from image local position and downsample factor
            # Assume that downsampled position = position * downsample factor

            # REMINDER:
            # Absolute position = ( relative position + offset of file as is ) / factor

            # For pixel in image
            # Get position of pixel in downsampled image
            # Convert to final image position
            # Insert into final array

            x, y = ascFile.index2coord(index, downscalefactor)
            imagex = (x + ((ascFile.minX - imageminX) / cellSize) / downscalefactor )
            imagey = (y + ((ascFile.minY - imageminY) / cellSize) / downscalefactor )
            try:
                thebigarray[int(imagex), int(imagey)] = int(mapValue(downsampled_image[x,y], (0, 4500), (0x0, 0x7fff)))
            except Exception:
                print("!!!")
                print(ascFile.data()[x*downscalefactor,y*downscalefactor])
                print(downsampled_image[x,y])
                print(index)
                print("!!!")
                raise
    thebigarray = np.rot90(thebigarray)
    img = png.from_array(thebigarray.tolist(), mode="L;16")
    img.save(outpath)


def multiple_images_mt(inpaths, outpath: Path) -> None:
    ascfiles = []
    for inpath in tqdm(inpaths, desc='Reading ASC files'):
        ascfiles.append(ASCFile(inpath))

    assert all([asc.cellSize == ascfiles[0].cellSize for asc in ascfiles])  # Check that cellsize is uniform across all files
    cellSize = ascfiles[0].cellSize
    assert all([asc.nCols == ascfiles[0].nCols for asc in ascfiles])  # Check that nCols is uniform across all files
    nCols = ascfiles[0].nCols
    assert all([asc.nRows == ascfiles[0].nRows for asc in ascfiles])  # Check that nRows is uniform across all files
    nRows = ascfiles[0].nRows

    minX = min([f.minX for f in ascfiles])
    maxX = max([f.maxX for f in ascfiles])
    minY = min([f.minY for f in ascfiles])
    maxY = max([f.maxY for f in ascfiles])

    offsetX = minX
    offsetY = minY
    size_x = int((maxX - minX) / cellSize)  # Assuming cellSize is the same for all files
    size_y = int((maxY - minY) / cellSize)  # Assuming cellSize is the same for all files

    thebigarray = np.zeros((size_x, size_y), dtype=np.int32)

    futs = []
    with ThreadPoolExecutor(max_workers=12) as tpe:
        for file in ascfiles:
            futs.append(tpe.submit(process_ascdata_to_map_array, file, thebigarray, nCols, nRows, cellSize, offsetX, offsetY))
        wait(futs)
    thebigarray = np.rot90(thebigarray)
    img = png.from_array(thebigarray.tolist(), mode='L;16')
    img.save(outpath)


def process_ascdata_to_map_array(ascfile: ASCFile, array: np.array, nCols: int, nRows: int, cellSize: int, offsetX: int, offsetY:int):
    for index in range(0, nCols * nRows):
        if index % 100000 == 0 and index != 0:
            print(ascfile.name+": "+str((nCols * nRows) / index))
        x, y = ASCFile.index2coord(index)
        localx = x + ((ascfile.minX - offsetX) / cellSize)
        localy = y + ((ascfile.minY - offsetY) / cellSize)
        try:
            array[int(localx), int(localy)] = int(mapValue(ascfile.data()[x, y], (0, 4500), (0x0, 0x7fff)))
        except Exception:
            print("!!!")
            print(ascfile.data()[x, y])


if __name__ == "__main__":
    chartreuse = [
        Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0925_6475_MNT_LAMB93_IGN69.asc"),
        Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0925_6500_MNT_LAMB93_IGN69.asc"),
        Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0900_6475_MNT_LAMB93_IGN69.asc"),
        Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0900_6500_MNT_LAMB93_IGN69.asc"),
    ]

    oneimg = [Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0925_6475_MNT_LAMB93_IGN69.asc")]

    isere = [Path("S:/Curiosités/IGN/BD ALTI - 38") / Path(filename) for filename in os.listdir(Path("S:/Curiosités/IGN/BD ALTI - 38"))]

    # one_image(Path("S:/Curiosités/IGN/BD ALTI - 38/BDALTIV2_25M_FXX_0900_6475_MNT_LAMB93_IGN69.asc"),Path("C:/Users/Antonin/Desktop/test2.tiff"))
    # multiple_images(files,Path("S:/Curiosités/IGN/BD ALTI - 38/beegoutput.tiff"))

    # multiple_images(sample(rhone_alpes, 10), Path("S:/Curiosités/IGN/rhonealpes.png"))

    multiple_images_scaled(isere, Path("S:/Curiosités/IGN/isere2.png"), downscalefactor=2)

    # multiple_images_scaled(chartreuse,Path("S:/Curiosités/IGN/charteusescale4.png"), downscalefactor=2)

    # os.chdir("F:/RGEALTI_MNT_1M_ASC_LAMB93_IGN69_D038_20210118")
    # multiple_images2([Path(asc) for asc in os.listdir(".")], Path("."))



    # multiple_images(os.listdir("."), "S:/Curiosités/IGN/RGEALTI_38.tiff")

    print("OK!")
