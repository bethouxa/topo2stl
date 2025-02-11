#!/bin/python3
# -*- coding: utf-8 -*-

import random
import re
from argparse import ArgumentParser

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


def mapValue(value: float, fromRange: tuple, toRange: tuple) -> float:
    # Correct span calculations
    from_min, from_max = fromRange
    to_min, to_max = toRange
    
    fromSpan = from_max - from_min
    toSpan = to_max - to_min

    # Calculate scale factor only once
    scale = toSpan / fromSpan

    # Single expression for conversion
    return (value - from_min) * scale + to_min


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

    @property
    def minAltitude(self) -> float:
        return self.data.min()

    @property
    def maxAltitude(self) -> float:  # Lazyload
        return self.data.max()

    @property
    def data(self) -> ndarray:
        if not hasattr(self, '_data'):
            # Load data only when accessed first time
            self._data = np.loadtxt(
                self._filepath,
                dtype=float,
                skiprows=6,
                encoding='utf-8'
            )
        return self._data
    
    def coord2index(self, x, y) -> int:
        return x * self.nCols + y

    def free_data(self) -> None:
        del self._data


def load_ascfiles(inpaths: list):
    ascfiles = []
    for inpath in tqdm(inpaths, desc="Reading ASC files"):
        ascfiles.append(ASCFile(inpath))
    return ascfiles


def process_files(inpaths: list, outpath: Path, downscalefactor: int = 1, transparency_on_empty: bool = False, modifier: Callable = None) -> None:
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

    interp_source = (modifier(0), modifier(4500))  # Max value of Z for a french map
    interp_target = (0, 0xffff)  # Max pixel color for given png mode (16 bits greyscale)

    image_size_X = int(math.ceil((image_max_X - image_min_X) / cellSize / downscalefactor))
    image_size_Y = int(math.ceil((image_max_Y - image_min_Y) / cellSize / downscalefactor))

    if transparency_on_empty:
        thebigarray = np.zeros((image_size_X*2, image_size_Y), dtype=np.int32)
        # *2 for color channel + alpha channel. Default value of 0 = transparent unless worked on later
    else:
        thebigarray = np.zeros((image_size_X, image_size_Y), dtype=np.int32)
        # No alpha channel

    for chunk in tqdm(ascfiles, desc="Processing file..."):
        # Downsample => for example, bring 1000x1000 image to 500x500 image
        downsampled_chunk = skimage.measure.block.block_reduce(chunk.data, block_size=downscalefactor, func=np.mean)
        downsampled_chunk = np.rot90(downsampled_chunk, k=3)
        
        # Apply both modifier and value mapping in a single vectorized operation
        # This approach provides several advantages:
        # 1. Performance: Utilizes NumPy's optimized C implementation for array operations
        # 2. Efficiency: Eliminates Python-level loops and multiple function calls per pixel
        # 3. Maintainability: Consolidates transformation logic into a single step
        # The lambda function performs the following:
        # - Applies the modifier function to each value
        # - Maps the modified value from the source range to the target range
        # - Converts the result to an integer (required for image pixel values)
        # This creates a complete pre-processed array ready for final placement
        mapped_chunk = np.vectorize(
            lambda value: int(mapValue(modifier(value), interp_source, interp_target))
        )(downsampled_chunk)

        # Note: While np.vectorize isn't as fast as native NumPy vectorized operations,
        # it's necessary here due to the complex custom transformation involving both
        # the user-defined modifier and the mapValue function. For most practical purposes,
        # this will still be significantly faster than processing each pixel individually.

        # Calculate offset positions
        offset_x = int((chunk.minX - image_min_X) / cellSize / downscalefactor)
        offset_y = int((chunk.minY - image_min_Y) / cellSize / downscalefactor)

        # Define target slice
        target_slice_x = slice(offset_x * 2 if transparency_on_empty else offset_x,
                               (offset_x + mapped_chunk.shape[0]) * 2 if transparency_on_empty else offset_x + mapped_chunk.shape[0])
        target_slice_y = slice(offset_y, offset_y + mapped_chunk.shape[1])

        # Assign values using numpy slicing
        if transparency_on_empty:
            thebigarray[target_slice_x:target_slice_x+1:2, target_slice_y] = mapped_chunk  # Color channel
            thebigarray[target_slice_x+1:target_slice_x+2:2, target_slice_y] = 0xffff      # Alpha channel
        else:
            thebigarray[target_slice_x, target_slice_y] = mapped_chunk

        # Free memory
        chunk.free_data()

    if transparency_on_empty:
        img = png.from_array(thebigarray.tolist(), mode="LA;16")
    else:
        img = png.from_array(thebigarray.tolist(), mode="L;16")
    img.save(outpath)

    print("Done!")
    print(str(outpath))


def benchmark_worker():
    from sys import argv
    inpath = Path(argv[1])
    infiles = [inpath/files for files in os.listdir(inpath) if files.endswith('.asc')]
    outfile = inpath/f'benchmark_{datetime.today().strftime("%A-%H%M%S")}.png'
    process_files(infiles, outfile, downscalefactor=2, transparency_on_empty=True, modifier=math.sqrt)


def benchmark():
    from cProfile import run
    from pstats import Stats
    from sys import argv
    statsFile = Path(argv[1])/f'profilestats-{datetime.today().strftime("%A-%H%M%S")}.cprofilestats'
    run("benchmark_worker()", str(statsFile))
    Stats(str(statsFile)).strip_dirs().sort_stats('cumtime').reverse_order().print_stats()
    print(str(statsFile))


def _make_paths(inpath: Path, outpath: Path = None, dsf: int = 1, transp: bool = True, modifier: Callable = lambda x: x):

    if inpath.is_dir():
        infiles = [inpath/files for files in os.listdir(inpath) if files.endswith('.asc')]
    else:
        infiles = [inpath]

    date = datetime.today().strftime('%Y-%m-%d')
    fname = f"{inpath.name}_scale{dsf}_{"transp_" if transp else ""}{modifier.__name__}_{date}.png"
    if outpath and not outpath.is_dir():
        assert outpath.parent.exists()
        outfile = outpath
    elif outpath and outpath.is_dir():
        assert outpath.parent.exists()
        outfile = outpath/fname
    else:
        if inpath.is_dir():
            outfile = inpath/fname
        else:
            outfile = inpath.parent/fname

    return (infiles, outfile)


def pow2(base: float):
    return base**2

if __name__ == "__main__":

    func_map = {
        "sqrt": math.sqrt,
        "log2": math.log2,
        "log10": math.log10,
        "sqre": pow2,
        "raw": lambda x: x
    }

    argparser = ArgumentParser(
        prog="MNT2Heightmap",
        description="Converts IGN BDALTI MNT maps (aka RGEALTI) to 16-bit greyscale heightmaps"
    )    
    argparser.add_argument('input_path', type=Path, help=
        "Directory or file to process. If the path is a directory, all '.mnt' files contained within will be processed."
    )
    argparser.add_argument('-o', '--output', type=Path, help=
        "Output path for the final image. Output file name will be autogenerated if a directory is provided."
    )
    argparser.add_argument('-d', '--downscale-factor', default=1, type=int, help=
        "Downscale factor: By how much the image will be shrunk in the X/Y axis relative to the source data"
    )
    argparser.add_argument('-m', '--modifier', type=str, choices=list(func_map.keys()), default='raw', help=
        "Modifier that will be applied to altitude values"
    ) 
    argparser.add_argument('-t', '--transparent', action='store_true', help=
        "Enable transparency on missing data. Warning: will double output image size."
    )
    args = argparser.parse_args()

    infiles,outpath = _make_paths(
        inpath=args.input_path,
        outpath=args.output,
        dsf=args.downscale_factor,
        modifier=func_map[args.modifier],
        transp=args.transparent
    )

    process_files(
        inpaths=infiles,
        outpath=outpath,
        downscalefactor=args.downscale_factor, 
        modifier=func_map[args.modifier],
        transparency_on_empty=args.transparent
    )
