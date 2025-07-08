#!/bin/python3
# -*- coding: utf-8 -*-

import re
from argparse import ArgumentParser

import numpy as np
from pathlib import Path

import skimage.measure.block
import os

import math
from numpy import ndarray
from tqdm import tqdm

from typing import Callable
import itertools
from datetime import datetime

import png
from PIL import Image,ImageFont,ImageDraw

# Utility functions


def mapValue(value, fromRange: tuple, toRange: tuple) -> float:
    # Figure out how 'wide' each range is
    fromSpan = fromRange[0] - fromRange[1]
    toSpan = toRange[0] - toRange[1]

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - fromRange[0]) / float(fromSpan)

    # Convert the 0-1 range into a value in the right range.
    return toRange[0] + (valueScaled * toSpan)


def get_drawn_text_bitmap(text: str, whitevalue: int, fontfam: str = 'arial', fontsize: int = 16) -> np.ndarray:
    font = ImageFont.truetype(fontfam, fontsize)
    left,top,right,bottom = font.getbbox(text)
    w, h = int(right - left), int(bottom - top)
    h *= 2
    image = Image.new('L', (w, h), 1)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font)
    arr = np.asarray(image)
    arr = np.where(arr, 0, 1)
    arr = arr[(arr != 0).any(axis=1)]
    white_on_black_arr = arr  * whitevalue
    return white_on_black_arr


# Class used to hold properties and accesses to the raw input files + couple computed properties


class ASCFile:
    nCols: int
    nRows: int
    minX: float
    minY: float
    rowcount: int
    cellSize: int
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
                            self.cellSize = int(float(re.search("[0-9.]+", line).group()))  # int(float()) because the input is a string repr of a decimal number so we can't int() directly
            except Exception:
                print("Cannot parse " + self.name)
                raise
            self.maxX = self.minX + (self.nCols * self.cellSize)
            self.maxY = self.minY + (self.nRows * self.cellSize)
            self.size = (self.maxX-self.minX, self.maxY-self.minY)

    def __str__(self) -> str:
        try:
            self._data
        except AttributeError:
            return f"ASCFile {self.name}, X {self.minX}-{self.maxX}, Y {self.minY}-{self.maxY}, unloaded"
        else:
            return f"ASCFile {self.name}, X {self.minX}-{self.maxX}, Y {self.minY}-{self.maxY}, Z {self.minAltitude}-{self.maxAltitude}"

    def __repr__(self) -> str:
        try:
            self._data
        except AttributeError:
            return f"ASCFile object {str(self._filepath)=}, data loaded: no, {self.nRows=}, {self.nCols=}, {self.cellSize=}, {self.minX=} {self.maxX=}, {self.minY=} {self.maxY=}"
        else:
            return f"ASCFile object {str(self._filepath)=}, data loaded: yes, {self.nRows=}, {self.nCols=}, {self.cellSize=}, {self.minX=} {self.maxX=}, {self.minY=} {self.maxY=}, {self.minAltitude=} {self.maxAltitude=}"

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
        try:
            return self._data
        except AttributeError:
            self._data = np.loadtxt(self._filepath, dtype=float, skiprows=6, encoding='utf-8')
            return self._data

    def coord2index(self, x, y) -> int:
        """Converts a (x,y) coordinate into a data index, usable for ASCFile.data"""
        return x * self.nCols + y

    def free_data(self) -> None:
        del self._data

    def simple_image(self, outpath, dsf: int, maxAlt: int = None):
        """Generates an image from the current ASCFile. Does not support transparency, modifiers, compositing, marking, etc, see process_files for that"""
        if maxAlt is None:
            maxAlt = self.maxAltitude
        imagedata_raw: ndarray = skimage.measure.block.block_reduce(self.data, block_size=dsf, func=np.mean).astype(np.int32)
        imagedata_interpd: ndarray = np.array([mapValue(imagedata_raw[i], (0,maxAlt),(0,0xffff)) for i in range(0,imagedata_raw.size)]).astype(np.int32)
        png.from_array(imagedata_interpd.tolist(), mode="L;16").save(outpath)


# Business functions


def process_files(inpaths: list, outpath: Path, max_alti: int, downscalefactor: int = 1, transparency_on_empty: bool = False, modifier: Callable = lambda x: x, mark_files: bool = False) -> None:
    """Main processing step. Takes a list of file paths, stitches them together and converts them into a single, 16-bit per color, grayscale PNG heightmap"""

    white_value = 0xffff  # Magic constant equal to white value for the image mode used (here L;16 -> 16 bits -> 0xffff)

    ascfiles = [ASCFile(inpath) for inpath in tqdm(inpaths, desc="Reading ASC files")]

    # Check some arbitrary stuff that makes our life a lot easier
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

    interp_source = (modifier(0), modifier(max_alti))  # Max value of Z for a french map
    interp_target = (0, white_value)  # Max pixel color for given png mode (16 bits greyscale)

    image_size_X = int(math.ceil((image_max_X - image_min_X) / cellSize / downscalefactor))
    image_size_Y = int(math.ceil((image_max_Y - image_min_Y) / cellSize / downscalefactor))

    # Allocate the array used to house the final image
    if transparency_on_empty:
        thebigarray = np.zeros((image_size_X*2, image_size_Y), dtype=np.int32)
        # *2 for color channel + alpha channel. Default value of 0 = transparent unless worked on later
    else:
        thebigarray = np.zeros((image_size_X, image_size_Y), dtype=np.int32)
        # No alpha channel

    for chunk in tqdm(ascfiles, desc=f"Processing...", unit="file"):
        # Double check that we don't get above the max altitude
        if chunk.maxAltitude > max_alti:
            print(f"Warning: input file {chunk.name} has portions going above the specified max altitude ({max_alti}) and will be truncated")
        # Downsample => for example, bring 1000x1000 image to 500x500 image
        chunk_data = skimage.measure.block.block_reduce(chunk.data, block_size=downscalefactor, func=np.mean)
        # Our final image is north up, but the chunks are west-up, fix that
        chunk_data = np.rot90(chunk_data, k=3)

        # Compute the offset of the chunk in the final image (position of top left/north-west/minimal-x-and-y corner)
        chunk_corner_in_final_x = int(((chunk.minX - image_min_X) / cellSize) / downscalefactor)
        chunk_corner_in_final_y = int(((chunk.minY - image_min_Y) / cellSize) / downscalefactor)

        for index in range(0, chunk_data.size):
            chunk_x, chunk_y = chunk.index2coord(index, downscalefactor)
            # Compute the positions of the "current pixel" in the final image
            # Position in final image = position in current chunk + position of corner of chunk in final image
            image_x = chunk_corner_in_final_x + chunk_x
            image_y = chunk_corner_in_final_y + chunk_y
            if mark_files and (chunk_x == 0 or chunk_y == 0 or chunk_x == chunk.nRows-1 or chunk_y == chunk.nCols-1):
                pixelColor = white_value
            else:
                pixelColor = int(mapValue(modifier(min(max_alti,chunk_data[chunk_x, chunk_y])), interp_source, interp_target))

            if transparency_on_empty:
                thebigarray[int(image_x*2),   int(image_y)] = pixelColor  # Color
                thebigarray[int(image_x*2+1), int(image_y)] = 0xffff  # Transparency
            else:
                thebigarray[int(image_x),     int(image_y)] = pixelColor

        if mark_files:
            filename_bitmap = get_drawn_text_bitmap(chunk.name, whitevalue=white_value)
            filename_width,filename_height=np.shape(filename_bitmap)
            # Paste the name of the file on the top left corner of its area with a hardcoded 2 px margin
            thebigarray[chunk_corner_in_final_x+2:chunk_corner_in_final_x+2+filename_width, chunk_corner_in_final_y+2:chunk_corner_in_final_y+2+filename_height] = filename_bitmap

        chunk.free_data()

    if transparency_on_empty:
        img = png.from_array(thebigarray.tolist(), mode="LA;16")
    else:
        img = png.from_array(thebigarray.tolist(), mode="L;16")
    img.save(outpath)


# Benchmark helpers


def benchmark_worker():
    """Runs a job with predetermined parameters for benchmarking purposes"""
    from sys import argv
    inpath = Path(argv[1])
    infiles = [inpath/files for files in os.listdir(inpath) if files.endswith('.asc')]
    outfile = inpath/f'benchmark_{datetime.today().strftime("%A-%H%M%S")}.png'
    process_files(infiles, outfile, downscalefactor=2, transparency_on_empty=True, modifier=math.sqrt)


def benchmark():
    """Runs a pre-determined benchmark worker and displays stats"""
    from cProfile import run
    from pstats import Stats
    from sys import argv
    statsFile = Path(argv[1])/f'profilestats-{datetime.today().strftime("%A-%H%M%S")}.cprofilestats'
    run("benchmark_worker()", str(statsFile))
    Stats(str(statsFile)).strip_dirs().sort_stats('cumtime').reverse_order().print_stats()
    print(str(statsFile))


# Helpers for commandline stuff


def _make_paths(inpath: Path | str, outpath: Path = None, dsf: int = 1, transp: bool = True, modifier: Callable = lambda x: x):
    """Takes "human-supplied" paths and turns them into a valid list of inputs + valid output path for process_files,
    allows the user to specify just about any (valid but not necessarily existing) path(s) and it will figure things out."""

    # Input: 3 cases: list of files, directory, single file (invalid paths will throw FileNotFoundException from the else clause).
    if re.fullmatch(str(inpath), '".*?"(,".*?")+'):
        good_infiles = [Path(f) for f in inpath.split(',')]
    elif Path(inpath).is_dir():
        good_infiles = [Path(inpath)/files for files in os.listdir(inpath) if files.endswith('.asc')]
        if len(good_infiles) == 0:
            raise FileNotFoundError("No files with .asc extension found in specified directory")
    else:
        good_infiles = [Path(inpath)]

    # Output: 4 cases: is valid file path, is directory path, is not specified, and name depends on whether input is a file or list of files
    date = datetime.today().strftime('%Y-%m-%d')
    if len(good_infiles) > 1:
        basename = f"{good_infiles[0].parent.name}_scale{dsf}_{"transp_" if transp else ""}{date}.png"
    else:
        basename = f"{good_infiles[0].name}_scale{dsf}_{"transp_" if transp else ""}{date}.png"

    if outpath:  # Outfile specified
        if outpath.parent.exists() and not outpath.exists():  # Valid file path
            good_outfile = outpath
        elif outpath.exists() and outpath.is_dir():  # Valid directory path
            good_outfile = outpath / basename
        else:
            raise FileNotFoundError(f"Path {outpath} is invalid.")
    else:  # Outfile not specified
        good_outfile = Path(basename)

    return good_infiles, good_outfile


if __name__ == "__main__":

    func_map = {
        "sqrt": math.sqrt,
        "log2": math.log2,
        "log10": math.log10,
        "sqre": lambda x: x**2,
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
    argparser.add_argument('-a', '--max-altitude', default=4500, type=int, help=
        "Forces a maximum altitude for the purposes of scaling the image"
    )
    argparser.add_argument('-e', '--mark-edges', action='store_true', help=
        "Mark the edges of individual input files"
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
        transparency_on_empty=args.transparent,
        max_alti=args.max_altitude,
        mark_files=args.mark_edges
    )
