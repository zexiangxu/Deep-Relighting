import numpy as np
import os
from PIL import Image


class ComputeLoader(object):
    def __init__(self, imgs, dirs):
        self.inImgs = imgs
        self.imgSize = (imgs[0].shape[0], imgs[0].shape[1])
        self.inDirs = np.reshape(dirs, (-1,3))
        self._make_In()

    def _make_In(self):
        inputData = np.zeros((self.imgSize[0], self.imgSize[1]) + (len(self.inDirs) * 3,), np.float32)
        for ii in range(len(self.inDirs)):
            inputData[:, :, ii * 3:ii * 3 + 3] = self.inImgs[ii].astype(np.float32) / 255.0 * 2.0 - 1.0
        dirMaps = np.ones((self.imgSize[0], self.imgSize[1]) + (len(self.inDirs) * 2,), np.float32) * self.inDirs[:,:2].reshape(-1).astype(np.float32)
        inputData = np.append(inputData, dirMaps, axis=2)
        self.inputData = inputData

    def saveCurOutputs(self, outputs, outFolder, **kwargs):
        if len(outputs) != len(self.curIds):
            print "outputs cannot be saved"
            return
        for ii, id in enumerate(self.curIds):
            lightId = id
            subMatFolder = outFolder + "/%s/%d" % ("Shape__0", 0)
            if not os.path.isdir(subMatFolder + "/inters"):
                os.makedirs(subMatFolder + "/inters")
            if kwargs.has_key("scale"):
                outImg = outputs[ii]*kwargs["scale"]
            else:
                outImg = outputs[ii]
            svimg = np.uint8(np.clip(outImg, 0, 1) * 255)
            if kwargs.has_key("flipOpt") and len(kwargs["flipOpt"]) == 2:
                flipOpt = kwargs["flipOpt"]
                if flipOpt[0] == 1:
                    svimg = np.flipud(svimg)
                if flipOpt[1] == 1:
                    svimg = np.fliplr(svimg)

            if kwargs.has_key("cropSize"):
                size = kwargs["cropSize"]
                svimg = svimg[:size[0], :size[1]]
            Image.fromarray(svimg).save(
                subMatFolder + "/inters/%d_%.3f_%.3f.png" % (lightId, self.Coefs[lightId][0], self.Coefs[lightId][1]))


    def setTest(self, testMode = "TestALL", **kwargs):
        if kwargs.has_key("relightDirs"):
            relightDirs = kwargs ["relightDirs"]
            self.relightDirs = np.reshape(relightDirs, (-1,3))
            self.Coefs = self.relightDirs[:,:2] * 0.5 + 0.5
            self.relightDirsMap = self.relightDirs[:,:2].reshape((-1,1,1,2)).astype(np.float32)
            self.total = len(self.relightDirs)
        else:
            print "Error: no relightDirs specified."
            self.testMode = None
            return False

        self.testMode = testMode
        if self.testMode == "TestALL":
            self.testcandIds = np.arange(self.total)

        if kwargs.has_key("selectTargets") and kwargs["selectTargets"] != None:
            self.testcandIds = np.reshape(kwargs["selectTargets"], -1)

        self.testcur_id = len(self.testcandIds) + 1

        print "TestSize:", self.testSize()
        return True

    def testSize(self):
        if self.testMode == None:
            return 0
        return len(self.testcandIds)

    def update_next_test_batch_Ids(self, batch_size, **kwargs):
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
        inputData = np.zeros((len(ids),) + self.imgSize + (len(self.inDirs) * 5 + 2,))
        inputData[:,:,:,:-2] = self.inputData
        inputData[:,:,:,-2:] = self.relightDirsMap[ids]

        return inputData


    def next_in_batch_fromIds(self):
        inputs = self.takeInputs(self.curIds)
        return inputs


class ComputeLoader_sep(ComputeLoader):
    def __init__(self, imgs, dirs):
        super(ComputeLoader_sep, self).__init__(imgs, dirs)

    def _make_In(self):
        inputImgs = np.zeros((self.imgSize[0], self.imgSize[1]) + (len(self.inDirs)*3,), np.float32)
        for ii in range(len(self.inDirs)):
            inputImgs[:, :, ii * 3:ii * 3 + 3] = self.inImgs[ii]

        self.inputData = [inputImgs, self.inDirs[:,:2].reshape(-1)]

    def saveCurOutputs(self, outputs, outFolder, **kwargs):
        if len(outputs) != len(self.curIds):
            print "outputs cannot be saved"
            return
        for ii, id in enumerate(self.curIds):
            lightId = id
            subMatFolder = outFolder
            if not os.path.isdir(subMatFolder ):
                os.makedirs(subMatFolder )
            if kwargs.has_key("scale"):
                outImg = outputs[ii]*kwargs["scale"]
            else:
                outImg = outputs[ii]
            svimg = np.uint8(np.clip(outImg, 0, 1) * 255)
            if kwargs.has_key("flipOpt") and len(kwargs["flipOpt"]) == 2:
                flipOpt = kwargs["flipOpt"]
                if flipOpt[0] == 1:
                    svimg = np.flipud(svimg)
                if flipOpt[1] == 1:
                    svimg = np.fliplr(svimg)
            if kwargs.has_key("cropSize"):
                size = kwargs["cropSize"]
                svimg = svimg[:size[0], :size[1]]
            Image.fromarray(svimg).save(
                subMatFolder + "/%d_%.3f_%.3f.png" % (lightId, self.Coefs[lightId][0], self.Coefs[lightId][1]))


    def setTest(self, testMode = "TestALL", **kwargs):
        if kwargs.has_key("relightDirs"):
            relightDirs = kwargs ["relightDirs"]
            self.relightDirs = np.reshape(relightDirs, (-1,3))
            self.Coefs = self.relightDirs[:,:2] * 0.5 + 0.5
            self.relightDirs = self.relightDirs
            self.total = len(self.relightDirs)
        else:
            print "Error: no relightDirs specified."
            self.testMode = None
            return False

        self.testMode = testMode
        if self.testMode == "TestALL":
            self.testcandIds = np.arange(self.total)

        if kwargs.has_key("selectTargets") and kwargs["selectTargets"] != None:
            self.testcandIds = np.reshape(kwargs["selectTargets"], -1)

        self.testcur_id = len(self.testcandIds) + 1

        print "TestSize:", self.testSize()
        return True

    def testSize(self):
        if self.testMode == None:
            return 0
        return len(self.testcandIds)

    def update_next_test_batch_Ids(self, batch_size, **kwargs):
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
        inputImgs = np.ones((len(ids),) + self.imgSize + (len(self.inDirs) * 3,))
        inputImgs[:] = self.inputData[0]

        inputDirs = np.ones((len(ids), 2*len(self.inDirs)))
        inputDirs[:] = self.inputData[1]
        inputDirs = inputDirs.reshape(-1, 1, 1, 2*len(self.inDirs)).astype(np.float32)

        outDirs = self.relightDirs[ids,:2].reshape(-1, 1, 1, 2).astype(np.float32)
        return [inputImgs, inputDirs, outDirs]


    def next_in_batch_fromIds(self):
        inputs = self.takeInputs(self.curIds)
        return inputs
