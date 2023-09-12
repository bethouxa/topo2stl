from pathlib import Path
import re
import stl
import numpy

offsetX = 0.0 # NE PAS CHANGER ENTRE LES CARTES
offsetY = 0.0 # IDEM 
verticalScale = 1.0
inputFile = Path("S:/Curiosités/IGN/BD ALTI - 38/new 17.asc")

constellation = []

def coord2index(x,y):
    return y*sizeX+x

def index2coord(index):
    return (index % sizeX,index // sizeX)



print("Lecture du fichier d'entrée...")

rowcount = 0
for line in open(inputFile,'r'):
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
        for digit in digits:
            constellation.append([colcount*cellsize+minX-offsetX, rowcount-7*cellsize+minY-offsetY, float(digit)*verticalScale])
            colcount += 1
print("Génération des faces du relief")

faces = []
for pointindex in range(0,(len(constellation)-sizeX)): # For every point except those on the last row
    x,y = index2coord(pointindex)
    faces.append([pointindex,coord2index(x+1,y),coord2index(x+1,y+1)]) # Upper triangle (right angle on top right)
    faces.append([pointindex,coord2index(x,y+1),coord2index(x+1,y+1)]) # Lower triangle (right angle on bottom left)

print ("Génération de la bordure du modèle...")

# ranges of __indexes__ of points.
modelPoints = (0,len(constellation))

for x in range(0,sizeX): # Top edge, y=0
    point = constellation[coord2index(x,0)]
    constellation.append([point[0],point[1],0])
    
for y in range(0,sizeY): # Left edge, x=0
    point = constellation[coord2index(0,y)]
    constellation.append([point[0],point[1],0])

for y in range(0,sizeY): # Right edge, x=max
    point = constellation[coord2index(sizeX-1,y)]
    constellation.append([point[0],point[1],0])

for x in range(0,sizeX): # Bottom edge, y=max
    point = constellation[coord2index(x,sizeY-1)]
    constellation.append([point[0],point[1],0])

# ranges of __indexes__ of points.
topEdgePoints =    (modelPoints[1],     modelPoints[1] + sizeX)
leftEdgePoints =   (topEdgePoints[1],   topEdgePoints[1] + sizeX)
rightEdgePoints =  (leftEdgePoints[1],  leftEdgePoints[1] + sizeX)
bottomEdgePoints = (rightEdgePoints[1], rightEdgePoints[1] + sizeX)

'''
constellation map example for 1000x1000 map:
modelpoints:      0 - 999 999 -> Map data
topedgepoints:    1 000 000 - 1 000 999 -> top edge of base
leftedgepoints:   1 001 000 - 1 001 999 -> Left edge
rightedgepoints:  1 002 000 - 1 002 999 -> Right edge
bottomedgepoints: 1 003 000 - 1 003 999 -> Bottom edge
'''

print ("Génération des bords du modèle...")

for x in range(0,sizeX-1): # Top edge, y=0
    p = coord2index(x,0)
    faces.append([p, topEdgePoints[0]+x, topEdgePoints[0]+x+1])
    faces.append([p, coord2index(x+1,0), topEdgePoints[0]+x+1])

for y in range(0,sizeY-1): # Left edge, x=0
    p = coord2index(0,y)
    faces.append([p, leftEdgePoints[0]+y, leftEdgePoints[0]+y+1])
    faces.append([p, coord2index(0,y+1), leftEdgePoints[0]+y+1])

for y in range(0,sizeY-1): # Right edge, x=max
    p = coord2index(sizeX-1,y)
    faces.append([p, rightEdgePoints[0]+y, rightEdgePoints[0]+y+1])
    faces.append([p, coord2index(sizeX-1,y+1), rightEdgePoints[0]+y+1])

for x in range(0,sizeX-1): # Bottom edge, y=max
    p = coord2index(x,sizeY-1)
    faces.append([p, bottomEdgePoints[0]+x, bottomEdgePoints[0]+x+1])
    faces.append([p, coord2index(x+1,sizeY-1), bottomEdgePoints[0]+x+1])

faces.append([topEdgePoints[0],rightEdgePoints[0],bottomEdgePoints[1]-1])
faces.append([topEdgePoints[0],bottomEdgePoints[1]-1,leftEdgePoints[1]-1])

print("Encodage du fichier STL...")

npvertices = numpy.array(constellation)
npfaces = numpy.array(faces)

from pprint import pprint as pp
pp(constellation)
pp(faces)

model = stl.mesh.Mesh(numpy.zeros(npfaces.shape[0],dtype=stl.mesh.Mesh.dtype))
for i, f in enumerate(npfaces):
    for j in range(3):
        model.vectors[i][j] = npvertices[f[j],:]

model.save("omg.stl")

print("OK!")

'''
constellation map:

0 - 999 999 -> Map data
1 000 000 - 1 000 999 -> top edge of base
1 001 000 - 1 001 999 -> Left edge
1 002 000 - 1 002 999 -> Right edge
1 003 000 - 1 003 999 -> Bottom edge




  ------> x
 |  ┌───┬───┬───┬───┬
 |  │ ╲ │ ╲ │ ╲ │ ╲ │
y|  ├───┼───┼───┼───┼
 |  │ ╲ │ ╲ │ ╲ │ ╲ │
 |  ├───┼───┼───┼───┼
\ / │ ╲ │ ╲ │ ╲ │ ╲ │




────────────────────────





print ("Génération des bords du modèle...")

for x in range(0,sizeX-1): # Top edge, y=0
    p = coord2index(x,0)
    faces.append([p, 1000000+x, 1000001+x])
    faces.append([p, coord2index(x+1,0), 1000001+x])

for y in range(0,sizeY-1): # Left edge, x=0
    p = coord2index(0,y)
    faces.append([p, 1001000+y, 1001001+y])
    faces.append([p, coord2index(0,y+1), 1001001+y])

for y in range(0,sizeY-1): # Right edge, x=max
    p = coord2index(sizeX-1,y)
    faces.append([p, 1002000+y, 1002001+y])
    faces.append([p, coord2index(sizeX-1,y+1), 1002001+y])

for x in range(0,sizeX-1): # Bottom edge, y=max
    p = coord2index(x,sizeY-1)
    faces.append([p, 1003000+x, 1003001+x])
    faces.append([p, coord2index(x+1,sizeY-1), 1003001+x])

faces.append([1000000,1000999,1001999])
faces.append([100200,1002999,1001999])



'''


