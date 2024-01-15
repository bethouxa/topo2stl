from PIL import Image
from pathlib import Path
import re
from sys import argv
from numpy import little_endian
import tifffile

inpath = Path("S:\Curiosités\IGN\BD ALTI - 38\BDALTIV2_25M_FXX_0900_6475_MNT_LAMB93_IGN69.asc")
outpath = Path("S:\Curiosités\IGN\BD ALTI - 38\BDALTIV2_25M_FXX_0900_6475_MNT_LAMB93_IGN69.png")
verticalScale = 1.0
minColorValue = -0xffff
maxColorValue = 0xffff
maxMapValue = 4900

def coord2index(x,y):
    return y*sizeX+x

def index2coord(index):
    return (index % sizeX,index // sizeX)

def mapValue2colorValue(mapvalue):
    mapvalueRelativeToMax = mapvalue / maxMapValue
    mappedValue = (minColorValue - maxColorValue) * mapvalueRelativeToMax + minColorValue
    return round(mappedValue)

def main():
    data = []
    
    rowcount = 0
    for line in open(inpath,'r'):
        rowcount += 1
        if rowcount == 1:
            sizeX = int(re.search("[0-9.]+",line).group())
        elif rowcount == 2:
            sizeY = int(re.search("[0-9.]+",line).group())
        elif rowcount == 3:
            minX = float(re.search("[0-9.]+",line).group())
        elif rowcount == 4:
            minY = float(re.search("[0-9.]+",line).group())
        elif rowcount == 5:
            cellsize = float(re.search("[0-9.]+",line).group())
        elif rowcount == 6:
            nodata_val = re.search("[0-9.]+",line).group()
        elif rowcount >= 7:
            colcount = 0
            digits = line.split()
            lineData = []
            for digit in digits:
                lineData.append(float(digit))
            data.append(lineData)
      
    img = Image.new('I',(sizeX,sizeY))
    for y in range(0,sizeY):
        for x in range(0,sizeX):
            mapvalue = mapValue2colorValue(data[y][x])
            img.putpixel((x,y),(mapvalue))

    img.save(Path(outpath))
      



if __name__ == "__main__":
    main()