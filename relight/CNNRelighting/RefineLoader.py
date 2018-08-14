import numpy as np
import os
from PIL import Image


# load training pairs from joint optimization dataset
class PerViewInLoaderSame(object):
    def __init__(self, inFolder):
        self.inFolder = inFolder
        
        self.__loadMapInfo_init()
        self.Dirs = np.load(self.dataFolder+"/Dirs.npy")
        self.Coefs = np.load(self.dataFolder+"/Coefs.npy")
        self.outIdsMap = np.load(inFolder + "/outIds.npy").astype(int)

        self.total = len(self.outIdsMap)

    def __loadMapInfo_init(self):
        with open(self.inFolder + "/mapInfo.txt", "r") as f:
            self.dataFolder = f.readline()[:-1]
            if self.dataFolder[:1] == ".":
                self.dataFolder = self.inFolder + self.dataFolder[1:]
            str = f.readline()[len("patchSize: "):-1]
            self.patchSize = [int(val) for val in str.split(" ")]
            tP = ()
            for ip in self.patchSize:
                tP += (ip,)
            self.patchSize = tP
            self.patchNum = int(f.readline()[len("patchNum: "):-1])

            self.inputNum = int(f.readline()[len("inputNum: "):-1])
            self.sampleNum = int(f.readline()[len("sampleNum: "):-1])
            self.shapeNum = int(f.readline()[len("shapeNum: "):-1])
            self.shapeNames = []
            for i in range(self.shapeNum):
                oneMat = f.readline()[:-1]
                self.shapeNames.append(oneMat)
            self.loadingFolder = f.readline()[:-1]
        self.perShape = self.sampleNum * self.patchNum
    def epochSize(self, **kwargs):
        return self.total

    def testSize(self):
        if self.testMode == None:
            return 0
        return len(self.testcandIds)
    
    def setTest(self, testMode = "TestALL", **kwargs):
        self.testMode = testMode
        if self.testMode == "TestALL":
            self.testcandIds = np.arange(self.total)
        self.testcur_id = len(self.testcandIds) + 1
        print "TestSize:", self.testSize()


    # create a sampling id-sequence of a whole epoch
    # try to make the data more uniformly distribute, each subEpoch contains equal number of images per shape
    def createSubEpochIds(self, bRan = True, subEpochNum = 1):
        # total patch num
        inputCandNum = self.shapeNum * self.patchNum


        # create a matrix, store each scene-patch's img ids.
        # each element is a img, each row is a total scene-patch

        # start create each scene-patch's random id array
        candPatchIds = []
        for i in range(inputCandNum):
            if bRan:
                pIds = i * self.sampleNum + np.random.permutation(self.sampleNum)
            else:
                pIds = i * self.sampleNum + np.arange(self.sampleNum)

            candPatchIds.append(pIds)
        idRect = np.reshape(candPatchIds, (inputCandNum, self.sampleNum))


        # bag num per scene-patch per sub epoch
        perSubEp = self.sampleNum / subEpochNum
        # an array to store the whole bag ids array, which will finally becomes the candidate outids array.
        idRange = []
        # start making each sub epoch
        for iS in range(subEpochNum):
            # take a sub epoch's img ids in the bag matrix
            if iS != subEpochNum - 1:
                subBags = idRect[:, iS * perSubEp:iS * perSubEp + perSubEp].reshape(-1)
            else:
                subBags = idRect[:, iS * perSubEp:].reshape(-1)
            # append the bag ids by random permute bag ids in this sub epoch
            if not bRan:
                subBagIds = np.arange(len(subBags))
            else:
                subBagIds = np.random.permutation(len(subBags))
            idRange = np.append(idRange, subBags[subBagIds])

        candIds = idRange.astype(int)



        return candIds
    
    def update_next_batch_Ids(self, batch_size, bRan=True, subEpochNum = 1):
        start = self.cur_id
        self.cur_id += batch_size
        if self.cur_id > self.total:

            self.candIds = self.createSubEpochIds(bRan, subEpochNum)
            # Start next epoch
            start = 0
            self.cur_id = batch_size
            assert batch_size <= self.total
        end = self.cur_id
        self.curIds = self.candIds[start:end]

    def update_next_test_batch_Ids(self, batch_size):
        if self.testMode == None:
            return None
        start = self.testcur_id
        self.testcur_id += batch_size
        if self.testcur_id >= len(self.testcandIds) + batch_size:

            start = 0
            self.testcur_id = batch_size
            assert batch_size <= len(self.testcandIds)
        end = min(self.testcur_id, len(self.testcandIds))
        self.curIds = self.testcandIds[start:end]

    def takeInputs(self, ids):

        inputIds = self.outIdsMap[ids, :self.inputNum].reshape(-1)
        inputImgsData = []#self.curInput

        shapeIds = ids / self.perShape
        subPatchIds = (ids % self.perShape) / self.sampleNum
        self.curInput = []
        for ii, id in enumerate(ids):
            subMatFolder = self.dataFolder + "/%s" % (self.shapeNames[shapeIds[ii]])
            inputImgs = np.load(subMatFolder + "/%d.npy" % (subPatchIds[ii]), "r")

            subIds = self.outIdsMap[id, :self.inputNum]

            oneInput = inputImgs[subIds].reshape((-1,)+self.patchSize + (3,))
            oneInput = np.dstack(oneInput)
            inputImgsData.append(oneInput)
        inputImgsData = np.reshape(inputImgsData, (-1,) + self.patchSize + (3 * self.inputNum,))

        inputDirs = self.Dirs[inputIds, :2].reshape(-1, 1, 1, 2*self.inputNum).astype(np.float32)
        outDirs = self.Dirs[self.outIdsMap[ids, self.inputNum], :2].reshape(-1,1,1,2).astype(np.float32)

        return [inputImgsData, inputDirs, outDirs]
    
    def takeOutputs(self, ids):
        ids = np.reshape(ids, -1).astype(int)
        outputs = []

        shapeIds = ids / self.perShape
        subPatchIds = (ids%self.perShape)/self.sampleNum
        self.curInput = []
        for ii,id in enumerate(ids):
            
            subMatFolder = self.dataFolder + "/%s" % (self.shapeNames[shapeIds[ii]])
            inputImgs = np.load(subMatFolder+"/%d.npy"%(subPatchIds[ii]), "r")

            oneOutput = inputImgs[self.outIdsMap[id, self.inputNum]].reshape(self.patchSize+(3,))
            outputs.append(oneOutput)
        # self.curInput = np.reshape(self.curInput, (-1,) + self.patchSize + (3 * self.inputNum,))
        outputs = np.reshape(outputs, (-1,) + self.patchSize + (3,))
        return outputs

    def next_inOut_batch_fromIds(self):
        outputs = self.takeOutputs(self.curIds)
        inputs = self.takeInputs(self.curIds)

        return inputs, outputs

# input mapper loader, containing all information in the mapper,
# it provides functions to load the input data
class PerViewInLoaderSep(object):
    def __init__(self, dataFolder):
        self.dataFolder = dataFolder
        self.__loadMapInfo_init()
        self.allInDirs = np.load(dataFolder+"/allInDirs.npy")
        self.outIdsMap = np.load(dataFolder + "/outIds.npy")
        self.allOutDirs = np.load(dataFolder + "/allOutDirs.npy")
        self.allOutCoefs = np.load(dataFolder + "/allOutCoefs.npy")
        self.outIdsMap = np.load(dataFolder + "/outIds.npy")
        self.cropMap = np.load(dataFolder+"/cropIds.npy")

        # find the actual data
        if self.loadingFolder != "":
            if self.loadingFolder[0] == ".":
                self.dataFolder = self.dataFolder + self.loadingFolder[1:]
            else:
                self.dataFolder = self.loadingFolder

        self.total = len(self.outIdsMap)
        print "%s: data contains %d training pairs."%(self.dataFolder, self.total)

        self.cur_id = self.total + 1
        self.candIds = np.arange(self.total)

        self.testMode = None
        self.testParam = None

    # load basic information
    def __loadMapInfo_init(self):
        with open(self.dataFolder + "/mapInfo.txt", "r") as f:
            self.outImgFolder = f.readline()[:-1]
            str = f.readline()[len("patchSize: "):-1]
            self.patchSize = [int(val) for val in str.split(" ")]
            tP = ()
            for ip in self.patchSize:
                tP += (ip,)
            self.patchSize = tP
            self.patchNum = int(f.readline()[len("patchNum: "):-1])

            self.inputNum = int(f.readline()[len("inputNum: "):-1])
            self.sampleNum = int(f.readline()[len("sampleNum: "):-1])
            self.shapeNum = int(f.readline()[len("shapeNum: "):-1])
            self.shapeNames = []
            for i in range(self.shapeNum):
                oneMat = f.readline()[:-1]
                self.shapeNames.append(oneMat)
            self.loadingFolder = f.readline()[:-1]

    def epochSize(self, **kwargs):
        return self.total

    def testSize(self):
        if self.testMode == None:
            return 0
        return len(self.testcandIds)


    def takeInputs(self, ids):

        inputIds = self.outIdsMap[ids, 0]
        inputImgsData = []
        inputDirs = self.allInDirs[inputIds, :,:2].reshape(-1, 1, 1, 2*self.inputNum).astype(np.float32)
        outDirs = self.allOutDirs[ids, :2].reshape(-1,1,1,2).astype(np.float32)


        for ii,id in enumerate(inputIds):
            subMatFolder = self.dataFolder + "/%s" % (self.shapeNames[id/self.patchNum])
            inputImgs = np.load(subMatFolder+"/inputs/%d.npy"%(id%self.patchNum))
            inputImgsData.append(inputImgs)

        inputImgsData = np.reshape(inputImgsData, (-1,) + self.patchSize + (3 * self.inputNum,)).astype(np.uint8)


        return [inputImgsData, inputDirs, outDirs]


class PerLightPatchDataLoader(object):
    def __init__(self, dataFolder, dataTerms = ["inters"], dataTypes = [np.uint8]):
        self.dataFolder = dataFolder
        self.dataTerms = dataTerms
        self.dataTypes = dataTypes
    def takeData(self, shapeIds, patchIds, outIds, outCoefs, cropMaps, patchSize):
        outData = []
        for term in self.dataTerms:
            outData.append([])

        for id in range(len(shapeIds)):
            for it, term in enumerate(self.dataTerms):
                data = np.asarray(Image.open(self.dataFolder + "/Shape__%d/%d/%s/%d_%.3f_%.3f.png" % (
                    shapeIds[id],
                    patchIds[id],
                    term,
                    outIds[id],
                    outCoefs[id][0], outCoefs[id][1])), np.uint8)


                patch = data

                outData[it].append(patch)
        for it in range(len(self.dataTerms)):
            outData[it] = np.reshape(outData[it], (-1,)+outData[it][0].shape).astype(self.dataTypes[it])
        return outData

    def saveCurOutputs(self, outFolder, alloutputs, shapeIds, outIds, outCoefs, bSingle = True, prefix = "", **kwargs):
        for it, term in enumerate(self.dataTerms):

            term+=prefix

            #if this is the only output data
            if bSingle == True:
                outputs = alloutputs
            else:
                outputs = alloutputs[it]

            #save each term
            for id in range(len(shapeIds)):
                subMatFolder = outFolder + "/Shape__%d/0/%s" %(shapeIds[id], term)
                if not os.path.isdir(subMatFolder):
                    os.makedirs(subMatFolder)
                if kwargs.has_key("scale"):
                    outImg = outputs[id] * kwargs["scale"]
                else:
                    outImg = outputs[id]
                if kwargs.has_key("gamma"):
                    outImg = np.power(outImg, kwargs["gamma"])
                svimg = np.uint8(np.clip(outImg, 0, 1) * 255)
                Image.fromarray(svimg).save(
                                subMatFolder + "/%d_%.3f_%.3f.png" % (outIds[id], outCoefs[id][0], outCoefs[id][1]))


# merge the input and output
class DataMerger(object):
    def __init__(self, dataLoader, auxLoaders):
        self.dataLoader = dataLoader
        self.auxLoaders = auxLoaders

        self.total = self.dataLoader.total
        self.cur_id = self.total + 1

    def epochSize(self, **kwargs):
        return self.total

    def testSize(self):
        if self.testMode == None:
            return 0
        return len(self.testcandIds)

    def setTest(self, testMode = "TestALL", **kwargs):
        self.testMode = testMode
        if self.testMode == "TestALL":
            self.testcandIds = np.arange(self.total)
        self.testcur_id = len(self.testcandIds) + 1
        print "TestSize:", self.testSize()



    def saveCurOutputs(self, alloutputs, outFolder, **kwargs):
        alloutputs.reverse()

        id =0
        for il, loader in enumerate(self.auxLoaders):
            bSingle = False

            #if only one output just take it, and let the loader know it is only itself
            if len(self.auxLoaders) == 1 and len(self.auxLoaders[0].dataTerms) == 1:
                outputs = alloutputs
                bSingle = True
            else:
            #otherwise, collect the corresponding outputs for the loader
                outputs = []
                for term in loader.dataTerms:
                    outputs.append(alloutputs[id])
                    id+=1


            ids = np.reshape(self.curIds, -1).astype(int)
            inputIds = self.dataLoader.outIdsMap[ids, 0]
            shapeIds = inputIds / self.dataLoader.patchNum
            outIds = self.dataLoader.outIdsMap[ids, 1]
            outCoefs = self.dataLoader.allOutCoefs[ids]

            curPre = ""
            if  kwargs.has_key("prefix"):
                if il < len(kwargs["prefix"]):
                    curPre = kwargs["prefix"][il]

            loader.saveCurOutputs(outFolder, outputs, shapeIds, outIds, outCoefs, bSingle, curPre, **kwargs)


    # create a sampling id-sequence of a whole epoch
    # try to make the data more uniformly distribute, each subEpoch contains equal number of images per shape
    def createSubEpochIds(self, bRan = True, subEpochNum = 1):
        # total patch num
        inputCandNum = self.dataLoader.shapeNum * self.dataLoader.patchNum


        # create a matrix, store each scene-patch's img ids.
        # each element is a img, each row is a total scene-patch

        # start create each scene-patch's random id array
        candPatchIds = []
        for i in range(inputCandNum):
            if bRan:
                pIds = i * self.dataLoader.sampleNum + np.random.permutation(self.dataLoader.sampleNum)
            else:
                pIds = i * self.dataLoader.sampleNum + np.arange(self.dataLoader.sampleNum)

            candPatchIds.append(pIds)
        idRect = np.reshape(candPatchIds, (inputCandNum, self.dataLoader.sampleNum))


        # bag num per scene-patch per sub epoch
        perSubEp = self.dataLoader.sampleNum / subEpochNum
        # an array to store the whole bag ids array, which will finally becomes the candidate outids array.
        idRange = []
        # start making each sub epoch
        for iS in range(subEpochNum):
            # take a sub epoch's img ids in the bag matrix
            if iS != subEpochNum - 1:
                subBags = idRect[:, iS * perSubEp:iS * perSubEp + perSubEp].reshape(-1)
            else:
                subBags = idRect[:, iS * perSubEp:].reshape(-1)
            # append the bag ids by random permute bag ids in this sub epoch
            if not bRan:
                subBagIds = np.arange(len(subBags))
            else:
                subBagIds = np.random.permutation(len(subBags))
            idRange = np.append(idRange, subBags[subBagIds])

        candIds = idRange.astype(int)

        return candIds



    def update_next_batch_Ids(self, batch_size, bRan=True, subEpochNum = 1):
        start = self.cur_id
        self.cur_id += batch_size
        if self.cur_id > self.total:
            # Finished epoch
            # self._epochs_completed += 1
            # Shuffle the data
            # self.candIds = np.arange(self.total)
            # perm = np.arange(self.total)
            # if bRan:
            #     np.random.shuffle(perm)
            # self.candIds = self.candIds[perm]
            print "creating subEpochs: ", subEpochNum
            self.candIds = self.createSubEpochIds(bRan, subEpochNum)
            # Start next epoch
            start = 0
            self.cur_id = batch_size
            assert batch_size <= self.total
        end = self.cur_id
        self.curIds = self.candIds[start:end]

    def update_next_test_batch_Ids(self, batch_size):
        if self.testMode == None:
            return None
        start = self.testcur_id
        self.testcur_id += batch_size
        if self.testcur_id >= len(self.testcandIds) + batch_size:

            start = 0
            self.testcur_id = batch_size
            assert batch_size <= len(self.testcandIds)
        end = min(self.testcur_id, len(self.testcandIds))
        self.curIds = self.testcandIds[start:end]

    def takeInputs(self, ids):
        ids = np.reshape(ids, -1).astype(int)
        inputs = self.dataLoader.takeInputs(ids)
        return inputs

    def takeOutputs(self, ids):
        ids = np.reshape(ids, -1).astype(int)
        inputIds = self.dataLoader.outIdsMap[ids, 0]
        if type(self.dataLoader.cropMap) != type(None):
            cropMaps = self.dataLoader.cropMap[inputIds]
        else:
            cropMaps = None
        shapeIds = inputIds / self.dataLoader.patchNum
        outIds = self.dataLoader.outIdsMap[ids, 1]
        outCoefs = self.dataLoader.allOutCoefs[ids]

        outputs = []

        for aux in self.auxLoaders:
            oneOut = aux.takeData(shapeIds, inputIds%self.dataLoader.patchNum, outIds, outCoefs, cropMaps, self.dataLoader.patchSize)
            outputs += oneOut

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def next_inOut_batch_fromIds(self):

        inputs = self.takeInputs(self.curIds)
        outputs = self.takeOutputs(self.curIds)

        return inputs, outputs

    def next_batch(self, batchSize, bRan):
        self.update_next_batch_Ids(batchSize, bRan)
        return self.next_inOut_batch_fromIds()

    def next_test_batch(self, batchSize):
        self.update_next_test_batch_Ids(batchSize)
        return self.next_inOut_batch_fromIds()