from relight.CNNRelighting.RelightingNetwork import *
from relight.CNNRelighting.SameDirsLoader import *
from relight.CNNRelighting.ParallelLoaderWrapper import *
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":



    candDegs = [30, 45, 60, 90]

    parser = argparse.ArgumentParser(description= "relight test data")
    parser.add_argument('deg', choices=candDegs, type=int, default=90, help="Specify the degree of the relighting cone. Corresponding weights and directions will be selected from weightFolder and dirFolder.")
    parser.add_argument('inputNum', choices=[3,4,5,6,7,8], default=5, type=int, help="Specify the number of input samples. Corresponding weights and directions will be selected from weightFolder and dirFolder.")
    parser.add_argument('-i', '--inputFolder', default = "../data/test/mapper_100_sameDir", help="The folder containing the test data.")
    parser.add_argument('-w', '--weightFolder', default = "../trained/refine", help = "The folder containing the trained weights.")
    parser.add_argument('-d', '--dirFolder', default = "../trained/dirs", help = "The folder containing the learned light directions.")
    parser.add_argument('-o', '--outFolder', default= "../out/test/refine", help = "Output folder.")
    parser.add_argument('-u', '--unSaveImage', action='store_false', help="A flag to disable saving the relit images and only test errors will be calculated.")
    parser.add_argument('-t', '--targets', metavar='id', type=int, nargs='*', default=[], help="If specified, only a subset of scenes will be tested.")

    args = parser.parse_args()

    testFolder = args.inputFolder
    weightFolder = args.weightFolder
    outFolder = args.outFolder
    bSave = args.unSaveImage
    dirFolder = args.dirFolder
    targets = args.targets
    deg = args.deg
    inputNum = args.inputNum

    if not os.path.isfile(weightFolder + "/deg%d_%d.index"%(deg, inputNum)):
        print "no weights for deg%d %d"%(deg, inputNum)
        exit()

    print "testing deg %d input %d"%(deg, inputNum)
    print "using weight from: %s"%(weightFolder)

    if len(targets) != 0:
        print "targets: ", targets
        outFolder += "_targets"

    outFolder += "/deg%d_%d"%(deg, inputNum)




    orgTestSet = SameDirsLoader_slow(testFolder, [0, deg])
    print "rangeNum: ", len(orgTestSet.rangeIds)


    inDirs = np.loadtxt(dirFolder + "/deg%d_%d.txt"%(deg, inputNum))
    allDirs = np.load(testFolder + "/Dirs.npy")[orgTestSet.rangeIds]
    dots = np.reshape(allDirs, (-1, 3)).dot(inDirs.T)
    inIds = np.argmax(dots, axis=0)

    print inIds
    if len(targets) != 0:
        orgTestSet.setTest(inIds=inIds, selectTargets=targets)
    else:
        orgTestSet.setTest(inIds=inIds)

    testSet = ParallelLoaderWrapper(orgTestSet)


    msCNN = RelightingNet(imgSize=(512, 512), inputNum=len(inIds))


    if not os.path.isdir(outFolder + "/out"):
        os.makedirs(outFolder + "/out")


    error = msCNN.Test(outFolder + "/out", testSet,
                       restorePath= weightFolder + "/deg%d_%d"%(deg, inputNum),
                       batchSize=1, bSave=bSave)
    np.savetxt(outFolder + "/error.txt", np.reshape(error, -1))
