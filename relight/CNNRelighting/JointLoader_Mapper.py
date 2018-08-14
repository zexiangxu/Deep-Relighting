import numpy as np
import os
from PIL import Image
# import sys
# sys.path.append("..")
# # from Common import *

#The loader that loads the joint optimization data. Each data unit is a npy file that contains the images of the same patch rendered with of all 1053 directional lights.
class JointLoader_mapper(object):
    def __init__(self, dataFolder, inputNum = 4):
        self.inFolder = dataFolder

        self.Dirs = np.load(dataFolder+"/Dirs.npy")
        self.Coefs = np.load(dataFolder+"/Coefs.npy")
        self.CropIds = np.load(dataFolder+"/cropIds.npy")
        self.MapIds = np.load(dataFolder + "/mapIds.npy")
        self.__loadMapInfo_init()

        self.total = len(self.shapeNames) * len(self.Dirs) * self.patchNum
        self.numPerShape = len(self.Dirs) * self.patchNum
        self.numPerPatch = len(self.Dirs)

        self.inputNum = inputNum


        self.candIds = []
        self.cur_id = 1
        self.subStart = None

        self.weightsHolder = np.ones((1))
        self.inDirsHolder = np.ones((1))


    def __loadMapInfo_init(self):
        with open(self.inFolder + "/mapInfo.txt", "r") as f:
            self.dataFolder = f.readline()[:-1]
            if self.dataFolder[:1] == ".":
                self.dataFolder = self.inFolder + self.dataFolder[1:]
            self.shapeNum = int(f.readline()[len("shapeNum: "):-1])
            str = f.readline()[len("patchSize: "):-1]
            self.patchSize = [int(val) for val in str.split(" ")]
            tP = ()
            for ip in self.patchSize:
                tP += (ip,)
            self.patchSize = tP
            self.patchNum = int(f.readline()[len("patchNum: "):-1])
            self.shapeNames = []
            for i in range(self.shapeNum):
                oneMat = f.readline()[:-1]
                self.shapeNames.append(oneMat)

    def saveCurOutputs(self, g_outputs, outFolder, **kwargs):

        outputs, softMaxWeights = g_outputs

        if len(outputs) != len(self.curIds):
            print "outputs cannot be saved"
            return

        bSaveImg = True
        if kwargs.has_key("bSaveImg"):
            if kwargs["bSaveImg"] == False:
                bSaveImg = False

        if not os.path.isdir(outFolder):
            os.makedirs(outFolder)

        if not os.path.isfile(outFolder + "/softWeights.txt"):
            np.savetxt(outFolder + "/softWeights.txt", softMaxWeights.reshape(self.inputNum, len(self.Dirs)))
            np.savetxt(outFolder + "/softMaxC.txt",
                       np.argmax(softMaxWeights.reshape(self.inputNum, len(self.Dirs)), axis=1), "%d")
            np.savetxt(outFolder + "/softMax.txt",
                       np.max(softMaxWeights.reshape(self.inputNum, len(self.Dirs)), axis=1))

        for ii, id in enumerate(self.curIds):
            baseMatId = id / self.numPerShape
            subPatchId = (id % self.numPerShape) / self.numPerPatch
            subId = id % self.numPerPatch


            if bSaveImg:

                subMatFolder = outFolder + "/%s/%d" % (self.shapeNames[baseMatId], subPatchId)
                if not os.path.isdir(subMatFolder + "/inters"):
                    os.makedirs(subMatFolder + "/inters")
                if kwargs.has_key("scale"):
                    outImg = outputs[ii]*kwargs["scale"]
                else:
                    outImg = outputs[ii]
                svimg = np.uint8(np.clip(outImg, 0, 1) * 255)
                if kwargs.has_key("cropSize"):
                    size = kwargs["cropSize"]
                    svimg = svimg[:size[0], :size[1]]
                Image.fromarray(svimg).save(
                    subMatFolder + "/inters/%d_%.3f_%.3f.png" % (subId, self.Coefs[subId][0], self.Coefs[subId][1]))

    #recalculate the candidate ids
    def __updateCands(self, batch_size, maxPatchPerBatch, bRan=True, subEpochNum = 1):
        #total patch num
        inputCandNum = self.shapeNum * self.patchNum
        #bag size: num img per bag
        perBag = batch_size / maxPatchPerBatch
        #bg num per scene-patch
        bagPerPatch = self.numPerPatch / perBag
        #total bag num
        bagNum = bagPerPatch * inputCandNum

        if bRan:
            #create a matrix, store each scene-patch's bag ids.
            #each element is a bag, each row is a total scene-patch
            bagRect = np.arange(bagNum).reshape((inputCandNum, bagPerPatch))
            #bag num per sub epoch
            perSubEp = bagPerPatch / subEpochNum
            #an array to store the whole bag ids array, which will finally becomes the candidate outids array.
            bagRange = []
            #start making each sub epoch
            for iS in range(subEpochNum):
                #take a sub epoch's bags' ids in the bag matrix
                if iS != subEpochNum - 1:
                    subBags = bagRect[:,iS*perSubEp:iS*perSubEp+perSubEp].reshape(-1)
                else:
                    subBags = bagRect[:,iS*perSubEp:].reshape(-1)
                #append the bag ids by random permute bag ids in this sub epoch
                bagRange = np.append(bagRange, subBags[np.random.permutation(len(subBags))])
        else:
            bagRange = np.arange(bagNum)

        bagRange = bagRange.astype(int)

        #all outids
        candIds = np.arange(bagNum*perBag)

        #create each scene-patch's random id array, which will be fetch through bagRange, that creates the final random ids in a sub epoch style
        candPatchIds = []
        for i in range(inputCandNum):

            pIds = np.random.permutation(self.numPerPatch)

            candPatchIds.append(pIds)


        #take each bag from bagRange, fetch the corresponding img id through candpatchIds
        for i in range(bagNum):
            patchId = bagRange[i] / bagPerPatch
            subBag = bagRange[i] % bagPerPatch

            candIds[i*perBag:i*perBag+perBag] = patchId * self.numPerPatch + candPatchIds[patchId][subBag * perBag: subBag * perBag+perBag]
        self.subStart = 0





        print len(candIds), self.total
        # np.savetxt("testids.txt", candIds)

        return candIds



    def update_next_batch_Ids(self, batch_size, bRan=True, maxPatchPerBatch = 10, subEpochNum = 1):
        start = self.cur_id
        self.cur_id += batch_size
        if self.cur_id > len(self.candIds):

            self.candIds = self.__updateCands(batch_size, maxPatchPerBatch, bRan, subEpochNum)


            start = 0
            self.cur_id = batch_size
            assert batch_size <= self.total
        end = self.cur_id
        self.curIds = self.candIds[start:end]

    def epochSize(self, batchSize = 50, maxPatchPerBatch = 10,  **kwargs):
        candIds = self.__updateCands(batchSize, maxPatchPerBatch)
        return len(candIds)


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
        return True





    def takeOutputs(self, ids):
        ids = np.reshape(ids, -1).astype(int)
        # imgs = []
        curPid = -1
        ppatchId = -1

        if type(self.curImgs) == type(None):
            fullMappers = []
            curImgs = []
            patchIds = []
            for id in ids:
                shapeId = id / self.numPerShape
                subPatchId = (id % self.numPerShape) / self.numPerPatch
                subId = id % self.numPerPatch
                patchId = id / self.numPerPatch
                subMatFolder = self.dataFolder + "/%s" % (self.shapeNames[shapeId])

                if patchId != ppatchId:
                    if not patchId in patchIds:

                        fullInterMapper = np.load(subMatFolder + "/%d.npy" % subPatchId, "r")
                        fullMappers.append(fullInterMapper)

                        patchIds.append(patchId)
                        curPid = len(patchIds) - 1
                    else:
                        curPid = patchIds.index(patchId)
                    ppatchId = patchId
                curImgs.append(np.asarray(fullMappers[curPid][self.MapIds[subId]]))
            imgs = np.reshape(curImgs, (-1,)+self.patchSize+(3,))

        else:
            cids = []
            subids = []
            for id in ids:
                patchId = id / self.numPerPatch
                subId = id % self.numPerPatch

                if patchId != ppatchId:
                    curPid = self.curpatchIds.index(patchId)
                    ppatchId = patchId
                cids.append(curPid)
                subids.append(subId)
            #     interImg = self.curImgs[cid, subId].reshape(self.patchSize+(3,))*0.5 + 0.5
            #     imgs.append(interImg)
            # imgs = np.reshape(imgs, (-1,) + interImg.shape)
            imgs = self.curImgs[cids, subids].reshape((-1,)+self.patchSize+(3,))#*0.5 + 0.5



        # print "i ", ids[0], cids

        return imgs

    def takeInputs(self, ids):
        ids = np.reshape(ids, -1).astype(int)
        imgs = []
        outdirs = []
        patchIds = []


        batchSelect = np.zeros((len(ids), len(ids)), np.float32)
        curPid = -1
        ppatchId = -1
        for ii,id in enumerate(ids):
            shapeId = id / self.numPerShape
            subPatchId = (id % self.numPerShape) / self.numPerPatch
            subId = id % self.numPerPatch
            patchId = id / self.numPerPatch
            subMatFolder = self.dataFolder + "/%s" % (self.shapeNames[shapeId])
            if patchId != ppatchId:
                if not patchId in patchIds:

                    fullInterImg = np.load(subMatFolder + "/%d.npy" % subPatchId, "r")
                    interImg = np.array(fullInterImg[self.MapIds])
                    if len(imgs) == 0:
                        imgs = interImg.reshape((1,)+interImg.shape)
                    else:
                        imgs = np.append(imgs, interImg.reshape((1,)+interImg.shape), 0)

                    patchIds.append(patchId)
                    curPid = len(patchIds) - 1
                else:
                    curPid = patchIds.index(patchId)
                ppatchId = patchId

            outdirs.append(self.Dirs[subId, :2])
            batchSelect[ii, curPid] = 1.0
        batchSelect = batchSelect[:,:len(patchIds)]


        candimgs = imgs.reshape((len(imgs), -1, self.patchSize[0]*self.patchSize[1]*3))
        self.curImgs = candimgs
        self.curpatchIds = patchIds
        outdirs = np.reshape(outdirs, (-1,1,1,2)).astype(np.float32)

        if self.weightsHolder.shape != (len(patchIds), 1, 1):
            self.weightsHolder = np.ones((len(patchIds), 1, 1), np.float32)
        weightsHolder = self.weightsHolder

        if self.inDirsHolder.shape != (len(ids),) + (1,1,1):
            self.inDirsHolder = np.ones((len(ids),) + (1,1,1), np.float32)
        inDirsHolder = self.inDirsHolder

        # print "i ", ids[0], self.curpatchIds

        return candimgs, self.Dirs[:,:2].reshape((1, -1, 2)).astype(np.float32), outdirs, batchSelect, weightsHolder, inDirsHolder

    def next_batch(self, batch_size, **kwargs):
        self.update_next_batch_Ids(batch_size, **kwargs)
        return self.next_inOut_batch_fromIds()



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


    def next_inOut_batch_fromIds(self):
        inputs = self.takeInputs(self.curIds)
        outputs = self.takeOutputs(self.curIds)
        return inputs, outputs

    def next_test_batch(self, batch_size):
        self.update_next_test_batch_Ids(batch_size)
        return self.next_inOut_batch_fromIds()









