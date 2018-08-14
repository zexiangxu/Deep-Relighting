import numpy as np
import os
from PIL import Image


class SameDirsLoader_slow(object):
    def __init__(self, dataFolder, degRange = [0, 90]):


        self.dataFolder = dataFolder

        self.Dirs = np.load(dataFolder + "/Dirs.npy")
        self.Coefs = np.load(dataFolder + "/Coefs.npy")
        self.__loadMapInfo_init()

        self.total = len(self.shapeNames) * len(self.Dirs)
        self.numPerShape = len(self.Dirs)

        ids0 = np.reshape(np.where(self.Dirs[:, 2] <= np.cos(np.deg2rad(degRange[0]))), -1)
        ids1 = np.reshape(np.where(self.Dirs[ids0, 2] > np.cos(np.deg2rad(degRange[1]))), -1)
        self.rangeIds = ids0[ids1]
        self.rangeIdsMap = np.zeros(self.numPerShape, int) - 1
        self.rangeIdsMap[self.rangeIds] = range(len(self.rangeIds))


    def __loadMapInfo_init(self):
        with open(self.dataFolder + "/mapInfo.txt", "r") as f:
            self.orgDataFolder = f.readline()[:-1]
            if self.orgDataFolder[:1] == ".":
                self.orgDataFolder = self.dataFolder + self.orgDataFolder[1:]
            self.shapeNum = int(f.readline()[len("shapeNum: "):-1])
            self.shapeNames = []
            for i in range(self.shapeNum):
                oneMat = f.readline()[:-1]
                self.shapeNames.append(oneMat)

    def saveCurOutputs(self, outputs, outFolder, **kwargs):
        if len(outputs) != len(self.curIds):
            print "outputs cannot be saved"
            return
        for ii, id in enumerate(self.curIds):

            baseMatId = id / self.numPerShape
            lightId = id % self.numPerShape
            subMatFolder = outFolder + "/%s/%d" % (self.shapeNames[baseMatId], 0)
            if not os.path.isdir(subMatFolder + "/inters"):
                os.makedirs(subMatFolder + "/inters")
            if kwargs.has_key("scale"):
                outImg = outputs[ii] * kwargs["scale"]
            else:
                outImg = outputs[ii]
            if kwargs.has_key("gamma"):
                outImg = np.power(outImg, kwargs["gamma"])
            svimg = np.uint8(np.clip(outImg, 0, 1) * 255)
            if kwargs.has_key("cropSize"):
                size = kwargs["cropSize"]
                svimg = svimg[:size[0], :size[1]]
            Image.fromarray(svimg).save(
                subMatFolder + "/inters/%d_%.3f_%.3f.png" % (lightId, self.Coefs[lightId][0], self.Coefs[lightId][1]))


    def testSize(self):
        if self.testMode == None:
            return 0
        return len(self.testcandIds)

    def setTest(self, testMode = "TestALL", **kwargs):
        if kwargs.has_key("inIds"):
            inIds = kwargs ["inIds"]
        else:
            print "Error: no in Ids specified."
            self.testMode = None
            return False

        self.testMode = testMode
        if self.testMode == "TestALL":
            self.testcandIds = []
            for id in range(self.shapeNum):
                self.testcandIds = np.append(self.testcandIds,
                                             np.arange(id * self.numPerShape, id * self.numPerShape + self.numPerShape)[self.rangeIds])
            self.testcandIds = self.testcandIds.astype(int)
        if kwargs.has_key("selectTargets"):
            targets = kwargs["selectTargets"]
            self.testcandIds = []
            for id in targets:
                self.testcandIds = np.append(self.testcandIds, range(id*self.numPerShape, id*self.numPerShape+self.numPerShape))
            self.testcandIds = self.testcandIds.astype(int)
        self.testcur_id = len(self.testcandIds) + 1

        self.curInput = None
        self.curData = None
        self.curInputIDs = inIds
        self.curInputShapeId = None

        self.nextSet = None

        print "TestSize:", self.testSize()
        return True

    def takeOutputs(self, ids):
        ids = np.reshape(ids, -1).astype(int)
        imgs = []
        tempSet = [self.curInputShapeId, self.curInput, self.curData]
        for id in ids:
            if self.curInputShapeId != id/self.numPerShape:

                if self.nextSet[0] == id/self.numPerShape:
                    [self.curInputShapeId, self.curInput, self.curData] = self.nextSet
                else:
                    print "Error takeData In: ", id / self.numPerShape, self.curInputShapeId
                    self._takeData(id/self.numPerShape)

            subId = id % self.numPerShape
            interImg = self.curData[self.rangeIdsMap[subId]]
            imgs.append(interImg)
        [self.curInputShapeId, self.curInput, self.curData] = tempSet
        imgs = np.reshape(imgs, (-1,) + interImg.shape)
        return imgs

    def takeInputs(self, ids):
        if self.curInputIDs == None:
            return None

        ids = np.reshape(ids, -1).astype(int)
        inputData = []
        inDirs = []
        outDirs = []
        tempSet = [self.curInputShapeId, self.curInput, self.curData]
        for id in ids:
            if self.curInputShapeId != id/self.numPerShape:
                if self.nextSet[0] == id/self.numPerShape:
                    [self.curInputShapeId, self.curInput, self.curData] = self.nextSet
                else:
                    print "Error takeData In: ", id / self.numPerShape, self.curInputShapeId
                    self._takeData(id/self.numPerShape)

            oneInput, inDir = self.curInput
            dirMaps = np.ones((1, 1)+ (2,),np.float32) * self.Dirs[id % self.numPerShape, :2].astype(np.float32)

            inputData.append(oneInput)
            inDirs.append(inDir)
            outDirs.append(dirMaps)
        [self.curInputShapeId, self.curInput, self.curData] = tempSet
        inputData = np.reshape(inputData, (-1,) + inputData[0].shape)
        inDirs = np.reshape(inDirs, (-1,) + inDirs[0].shape)
        outDirs = np.reshape(outDirs, (-1,) + outDirs[0].shape)
        # print inputData.shape, inputData.dtype
        # print inDirs.shape, inDirs.dtype
        # print outDirs.shape, outDirs.dtype
        return [inputData, inDirs, outDirs]


    def _takeData(self, shapeId):
        self.curData = []
        subMatFolder = self.orgDataFolder + "/%s/%d" % (self.shapeNames[shapeId], 0)
        for i in self.rangeIds:
            interImg = np.asarray(Image.open(subMatFolder + "/inters/%d_%.3f_%.3f.png" % (
                i,
                self.Coefs[i][0], self.Coefs[i][1])), np.uint8)
            self.curData.append(interImg)
        self.curData = np.reshape(self.curData, (-1,) + interImg.shape)
        self.curInputShapeId = shapeId
        print "taken", self.curInputShapeId

        if self.curInputIDs != None:
            self.curInput = np.zeros((interImg.shape[0], interImg.shape[1])+ (len(self.curInputIDs)*3,),np.uint8)
            for ii,id in enumerate(self.curInputIDs):
                self.curInput[:,:,ii*3:ii*3+3] = self.curData[id]
            dirs = self.Dirs[self.rangeIds[self.curInputIDs],:2].reshape(-1).astype(np.float32)
            dirMaps = np.ones((1, 1)+ (len(self.curInputIDs)*2,),np.float32) * dirs
            self.curInput = [self.curInput, dirMaps]


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

        if self.curInputShapeId != self.curIds[0] / self.numPerShape:
            if self.nextSet != None:
                [self.curInputShapeId, self.curInput, self.curData] =  self.nextSet
            else:
                print "takeData In: ", self.curIds[0] / self.numPerShape, self.curInputShapeId
                self._takeData(self.curIds[0] / self.numPerShape)

        if self.curInputShapeId != self.curIds[-1] / self.numPerShape:
            self.tempSet = [self.curInputShapeId, self.curInput, self.curData]
            # print "next takeData In: ", self.curIds[-1] / self.numPerShape, self.curInputShapeId
            self._takeData(self.curIds[-1] / self.numPerShape)

            self.nextSet = [self.curInputShapeId, self.curInput, self.curData]
            # print "next0: ", self.nextSet[0], self.curInputShapeId
            [self.curInputShapeId, self.curInput, self.curData] = self.tempSet
            # print "next0: ", self.nextSet[0], self.curInputShapeId


    def next_inOut_batch_fromIds(self):
        outputs = self.takeOutputs(self.curIds)
        inputs = self.takeInputs(self.curIds)
        return inputs, outputs

    def next_test_batch(self, batch_size):
        self.update_next_test_batch_Ids(batch_size)
        self.next_inOut_batch_fromIds()
