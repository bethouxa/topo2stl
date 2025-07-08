# Topo2STL

Converts MNT topographical data (like those provided by IGN RGEALTI/BDALTI services) into greyscale heightmaps for use in 3D modeling software. The output images are greyscale PNGs with a resolution of 16 bit per channel which allow for a precision that will probably exceed the precision of your input data (0.13 m resolution).

Can take arbitrarily (im)precise maps and downscale as needed to fit any area/processing capabilities.

## Usage

```text
MNT2Heightmap [-h] [-o OUTPUT] [-d DOWNSCALE_FACTOR] [-m {sqrt,log2,log10,sqre,raw}] [-t] input_path

input_path
    Directory or file to process.
    If the path is a directory, all files with the ".mnt" extension contained within will be processed.

-h, --help
    show this help message and exit

-o output_path, --output output_path
    Output path for the final image
    If a directory is provided, the output file will be saved within using an auto-generated name.

-d downscale_factor, --downscale-factor downscale_factor
    Downscale factor: by how much the image will be shrunk in the X/Y axis relative to the source data. 
    Default is 1 (1 input data point = 1 px)

-m {sqrt,log2,log10,sqre,raw}, --modifier {sqrt,log2,log10,sqre,raw}
    Modifier that will be applied to altitude values. Default is 'raw' (color will be linearly proportional to altitude)

-t, --transparent
    Enable transparency on missing data. Warning: will double output image size.
    
-a max_altitude, --max-altitude max_altitude
    Forces a maximum altitude for the purposes of scaling the image
    Default is 4500 because the developer is French and has never seen a mountain taller than ~4410m.
  
-e, --mark-files      
    Mark the edges of individual input files and displays the original filename in the output image
```

## Output images information

### Format

Output images are PNGs with a single 16-bit channel (L;16 mode), plus an additional alpha channel (LA;16 mode) when -t is specified.
Pixel brightness is proportional to the altitude after modifiers are applied (see below), with 0x0 being 0m (according
to the input files, usually average sea level) and 0xFFFF being the MAX_ALTITUDE meters (by default 4500).
If source data is missing, the resulting color is 0x0. If the alpha channel is enabled with -t, the resulting pixel will additionally be transparent.

### Modifiers

By default (or with '-m raw') output image color is lineraly proportional to altitude, but the "linear" part can be
modified using the -m parameter. This is useful when the output is used for cosmetic purposes like in a 3D printing pipeline.

### Output image size

Image size in px is the source data maximum width times maximum height, divided by downscale factor (-d parameter). Ex: If the downscale factor is 2, 1 pixel represents a 2*2 "square" of source data.

## Developer's notes

Altitude data is lazyloaded: when an ASCFile object is instanced, only metadata is loaded and the full file is only read at the first call to ASCFile.data, ASCFile.minAltitude or ASCFile.maxAltitude.

For some ungodly reason the IGN MNT files are "west-up": the "top line" of the file represents the "leftmost" column of a traditional map. Note the rot90 call before incorporating the map data to the assembled image data.

Remember to call `ASCFile.free_data()` when working with big areas at small resolutions to avoid excessive RAM usage.

Primer on how to apply the heightmap to a mesh in blender: https://youtu.be/7-cGwA65hRA?si=4fc--g_BbEew7ysP&t=86
