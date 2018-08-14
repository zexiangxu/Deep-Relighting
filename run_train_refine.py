import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

sys.path.append("..")
from relight.CNNRelighting.RelightingNetwork import *
from relight.CNNRelighting.ParallelLoaderWrapper import *
from relight.CNNRelighting.RefineLoader import *
from relight.CNNRelighting.MultiDataLoader import *
import argparse

if __name__ == "__main__":


    candDegs = [45, 90]

    parser = argparse.ArgumentParser(description= "tran joint optimization")
    parser.add_argument('deg', choices=candDegs, type=int, default=90, help="Specify the degree of the relighting cone. Corresponding weights and directions will be selected from weightFolder and dirFolder.")
    parser.add_argument('inputNum', choices=[4,5], default=5, type=int, help="Specify the number of input samples. Corresponding weights and directions will be selected from weightFolder and dirFolder.")
    parser.add_argument('-i', '--dataFolder', default = "../data/train", help="The folder containing the joint training data (.npys).")
    parser.add_argument('-o', '--outFolder', default= "../out/refine", help = "Output folder.")
    parser.add_argument('-j', '--jointTrainedFolder', default= "../trained/joint", help = "Output folder.")
    parser.add_argument('-d', '--dirFolder', default = "../trained/dirs", help = "The folder containing the learned light directions.")

    args = parser.parse_args()

    dataFolder = args.dataFolder
    outFolder = args.outFolder
    deg = args.deg
    inputNum = args.inputNum
    jointTrainedFolder = args.jointTrainedFolder
    dirFolder = args.dirFolder


    if not os.path.isfile(jointTrainedFolder + "/deg%d_%d.index" % (deg, inputNum)):
        print "no jointly trained weights for deg%d %d" % (deg, inputNum)
        exit()

    print "refinement for deg: %d, inputNum: %d" % (deg, inputNum)
    print "output to: ", outFolder
    print "loading from: ", dataFolder
    print "joint weights from: ", jointTrainedFolder
    print "fix directions from: ", dirFolder


    orgTrainingSets = []

    trainFolder1 = "%s/joint_npy/mapper_Multi_train_500_deg%d_ranI%d"%(dataFolder, deg, inputNum)  # "/media/zexiang/ExtraSSD/data/relighting/joint_map_test_500"#"/media/zexiang/ExtraData/data/comet/mapper_Multi_500_deg45_ranI4"#
    if not os.path.isdir(trainFolder1):
        print  "No corresponding joint data for deg %d input %d"%(deg, inputNum)
        exit()
    orgTrainingSet1 = PerViewInLoaderSame(trainFolder1)
    orgTrainingSets.append(orgTrainingSet1)

    trainFolder2 = "%s/SepIn/mapper_Multi_5000_deg%d_ranI%d"%(dataFolder, deg, inputNum)  # "/media/zexiang/ExtraData/data/comet/mapper_Multi_5000_deg45_ranI4"#
    intersFolder2 = "%s/SepOut/Shape_Multi_5000_deg45_patch"%dataFolder
    if not os.path.isdir(trainFolder2) or not os.path.isdir(intersFolder2):
        print  "No corresponding refine data for deg %d input %d"%(deg, inputNum)
        exit()

    trainInput2 = PerViewInLoaderSep(trainFolder2)
    trainIntersLoader2 = PerLightPatchDataLoader(intersFolder2)
    orgTrainingSet2 = DataMerger(trainInput2, [trainIntersLoader2])
    orgTrainingSets.append(orgTrainingSet2)

    if deg >= 60:
        trainFolder3 = "%s/SepIn/mapper_Multi_5000_deg%d_ranI%d_45_60" % (dataFolder,
            deg, inputNum)  # "/media/zexiang/ExtraData/data/comet/mapper_Multi_5000_deg45_ranI4"#
        intersFolder3 = "%s/SepOut/Shape_Multi_5000_deg45_60_patch"%dataFolder

        if not os.path.isdir(trainFolder3) or not os.path.isdir(intersFolder3):
            print  "No corresponding refine data for deg %d input %d"%(deg, inputNum)
            exit()
        trainInput3 = PerViewInLoaderSep(trainFolder3)
        trainIntersLoader3 = PerLightPatchDataLoader(intersFolder3)
        orgTrainingSet3 = DataMerger(trainInput3, [trainIntersLoader3])
        orgTrainingSets.append(orgTrainingSet3)

    if deg == 90:
        trainFolder4 = "%s/SepIn/mapper_Multi_5000_deg%d_ranI%d_60_90" % (dataFolder,
            deg, inputNum)
        intersFolder4 = "%s/SepOut/Shape_Multi_5000_deg60_90_patch"%dataFolder
        if not os.path.isdir(trainFolder4) or not os.path.isdir(intersFolder4):
            print  "No corresponding refine data for deg %d input %d"%(deg, inputNum)
            exit()

        trainInput4 = PerViewInLoaderSep(trainFolder4)
        trainIntersLoader4 = PerLightPatchDataLoader(intersFolder4)
        orgTrainingSet4 = DataMerger(trainInput4, [trainIntersLoader4])
        orgTrainingSets.append(orgTrainingSet4)

    testFolder = "%s/joint_npy/mapper_Multi_test_100_deg%d_ranI%d"%(dataFolder, deg, inputNum)


    outFolder = outFolder+"/deg%d_I%d"%(deg, inputNum)


    batchSize = 50

    msCNN = RelightingNet(imgSize=(128, 128), inputNum=inputNum)
    print "loading training"


    orgTrainingSet = MultiDataLoader(orgTrainingSets)

    trainingSet = ParallelLoaderWrapper(orgTrainingSet)

    print "loading test"
    orgTestSet = PerViewInLoaderSame(testFolder)
    orgTestSet.setTest()
    testSet = ParallelLoaderWrapper(orgTestSet)



    msCNN.TrainTest_continue(outFolder,
                             trainingSet, testSet, 0.0001, batchSize=batchSize, subEpochNum = 10, testBatch=batchSize,
                             nEpochs=80,
                             restorePath=jointTrainedFolder+"/deg%d_%d"%(deg,inputNum),
                              saveTestFunc=testSet.saveCurOutputs, bSaveTest=False)


