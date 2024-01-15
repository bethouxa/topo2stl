from PIL import Image,ImageDraw
from pathlib import Path
import re
from sys import argv
from os import listdir, path
from math import floor

from numpy.core.fromnumeric import sort


class ASCFile:
    def __init__(self, path) -> None:
        self.data = []
        rowcount = 0
        for line in open(path, 'r'):
            rowcount += 1
            if rowcount == 1:
                self.sizeX = int(re.search("[0-9.]+", line).group())
            elif rowcount == 2:
                self.sizeY = int(re.search("[0-9.]+", line).group())
            elif rowcount == 3:
                self.minX = float(re.search("[0-9.]+", line).group())
            elif rowcount == 4:
                self.minY = float(re.search("[0-9.]+", line).group())
            elif rowcount == 5:
                self.cellsize = float(re.search("[0-9.]+", line).group())
            elif rowcount == 6:
                pass
            elif rowcount >= 7:
                digits = line.split()
                line_data = []
                for digit in digits:
                    line_data.append(float(digit))
                self.data.append(line_data)

    def coord2index(self, x, y) -> int:
        return y * self.sizeX + x

    def index2coord(self, index) -> tuple:
        return index % self.sizeX, index // self.sizeX

    @staticmethod
    def map_value_to_8b_color_value(map_value) -> int:
        _maxColorValue = 0xff
        _maxMapValue = 4100
        mapvalue_relative_to_max = map_value / _maxMapValue
        mapped_value = _maxColorValue * mapvalue_relative_to_max
        return round(mapped_value)

    @staticmethod
    def map_value_to_fmode_color_value(map_value) -> int:
        _maxColorValue = 0xffffffff
        _maxMapValue = 4100
        mapvalue_relative_to_max = map_value / _maxMapValue
        mapped_value = float(_maxColorValue * mapvalue_relative_to_max)
        return round(mapped_value)

    def get_altitude(self, x, y) -> float:
        return self.data[x][y]

    def get_color_value(self, x, y) -> int:
        return self.map_value_to_8b_color_value(self.get_altitude(x, y))


def one_image_8b() -> None:
    asc = ASCFile(argv[1])
    img = Image.new('RGB', (asc.sizeX, asc.sizeY))
    for y in range(0, asc.sizeY):
        for x in range(0, asc.sizeX):
            color = asc.get_color_value(x, y)
            img.putpixel((x, y), (color, color, color))
    img.save(Path(outpath))


def one_image_f_mode() -> None:
    asc = ASCFile(argv[1])
    img = Image.new('F', (asc.sizeX, asc.sizeY))
    imgDraw = ImageDraw.Draw(img,'F')
    for y in range(0, asc.sizeY):
        for x in range(0, asc.sizeX):
            color = asc.get_color_value(x, y)
            imgDraw.point([x, y], (color, color, color))
    img.save(Path(outpath))


def stitch_image(in_path) -> None:
    asc_filenames = list(filter(lambda f: path.splitext(f)[1] == ".asc", listdir(in_path)))
    asc_files = []
    for asc_filename in asc_filenames:
        asc_files.append(ASCFile(path.join(in_path, asc_filename)))
    global_min_x = floor(min([asc.minX for asc in asc_files]))
    global_min_y = floor(min([asc.minY for asc in asc_files]))

    total_x = sum([asc.sizeX for asc in asc_files])
    total_y = sum([asc.sizeY for asc in asc_files])

    img = Image.new('RGB', (total_x, total_y))

    for asc_file in asc_files:
        print(".", end="")
        pxoffset_x = floor((asc_file.minX - global_min_x) / asc_file.cellsize)
        pxoffset_y = floor((asc_file.minY - global_min_y) / asc_file.cellsize)
        for y in reversed(range(0, asc_file.sizeY)):
            for x in range(0, asc_file.sizeX):
                color = asc_file.get_color_value(y, x)
                img.putpixel((x + pxoffset_x, y + pxoffset_y), (color, color, color))

    img.save(in_path / "output.png")


if __name__ == "__main__":
    stitch_image(Path("""S:\Curiosités\IGN\BD ALTI - 38"""))
