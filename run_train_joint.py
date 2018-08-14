
import argparse
from relight.CNNRelighting.RelightingNetwork import *
from relight.CNNRelighting.JointLoader_Mapper import *
from relight.CNNRelighting.ParallelLoaderWrapper import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":


    sfaB = 6.0

    candDegs = [30, 45, 60, 90]

    parser = argparse.ArgumentParser(description= "tran joint optimization")
    parser.add_argument('deg', choices=candDegs, type=int, default=90, help="Specify the degree of the relighting cone. Corresponding weights and directions will be selected from weightFolder and dirFolder.")
    parser.add_argument('inputNum', choices=[3,4,5,6,7,8], default=5, type=int, help="Specify the number of input samples. Corresponding weights and directions will be selected from weightFolder and dirFolder.")
    parser.add_argument('-i', '--dataFolder', default = "../data/train/joint_npy", help="The folder containing the joint training data (.npys).")
    parser.add_argument('-o', '--outFolder', default= "../out/joint", help = "Output folder.")


    args = parser.parse_args()

    dataFolder = args.dataFolder
    outFolder = args.outFolder

    deg = args.deg
    inputNum = args.inputNum



    print "joint optimization for deg: %d, inputNum: %d"%(deg, inputNum)
    print "output to: ", outFolder
    print "loading from: ", dataFolder

    candMap = {30: 257, 45: 541, 60: 805, 90: 1053}
    batchSizeMap = {30: 64, 45: 72, 60: 92, 90: 105}
    subEpochNumMap = {30: 8, 45: 6, 60: 7, 90: 10}
    maxPatchPerBatchMap = {30: 4, 45: 4, 60: 4, 90: 3}

    subEpochNum = subEpochNumMap[deg]
    maxPatchPerBatch = maxPatchPerBatchMap[deg]
    batchSize = batchSizeMap[deg]

    trainFolder = dataFolder+"/Joint_Multi_train_500_deg%d_mapper"%deg  # "/media/zexiang/ExtraSSD/data/relighting/joint_map_test_500"
    testFolder = dataFolder+"/Joint_Multi_test_100_deg%d_mapper"%deg

    outFolder = outFolder+"/Joint_500_deg%d_I%d_q%.1f_midExpThick"%(deg, inputNum, sfaB)

    msCNN = RelightingJoint(imgSize=(128, 128), inputNum=inputNum, candNum=candMap[deg], sfaB=sfaB)
    print "loading training"
    orgTrainingSet = JointLoader_mapper(trainFolder, inputNum=inputNum,)
    trainingSet = ParallelLoaderWrapper(orgTrainingSet)

    print "loading test"
    orgTestSet = JointLoader_mapper(testFolder, inputNum=inputNum)
    orgTestSet.setTest()
    testSet = ParallelLoaderWrapper(orgTestSet)



    msCNN.TrainTest_continue(outFolder,
                    trainingSet, testSet, 0.0001, batchSize=batchSize, maxPatchPerBatch=maxPatchPerBatch,
                    subEpochNum=subEpochNum, testBatch=60,
                    nEpochs=80,
                    testMaxSize=105300, saveTestFunc=testSet.saveCurOutputs, bSaveTest=False)


