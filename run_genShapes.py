from relight.shapes.CreateShapes import *
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= "relight test data")
    parser.add_argument('-o', '--outFolder', default= "../out/genShapes", help = "Output folder.")
    parser.add_argument('-n', '--numShapes', default=100, type=int, help = "Number of shapes.")
    args = parser.parse_args()

    outFolder = args.outFolder
    numShapes = args.numShapes
    createVarObjShapes(outFolder, numShapes)

