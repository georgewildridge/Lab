#loading multiple images from one directory
import os, sys
from PIL import Image

# Open a file
path = '/Users/George/Desktop/BerkelyBenchmark/BSR/BSDS500/data/images/train/'
dirs = os.listdir( path )
valid_images = [".jpg"]
# This would print all the files and directories
for file in dirs:
    ext = os.path.splitext(file)[1]
    if ext.lower() not in valid_images:
        continue
    Image.open(path + file)