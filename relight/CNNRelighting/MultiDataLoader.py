import numpy as np


# A loader that can merge multiple data loader into one.
class MultiDataLoader(object):
    def __init__(self, loaders = []):
        self.loaders = loaders

        self.total = 0
        self.IdsBorder = []
        for loader in loaders:
            self.IdsBorder.append(self.total)
            self.total += loader.total


        print "data contains %d training pairs. in %d sep loaders" % (self.total, len(self.loaders))
        self.cur_id = self.total + 1
        self.candIds = np.arange(self.total)

        self.testMode = None
        self.testParam = None


    def epochSize(self, **kwargs):
        return self.total

    def testSize(self):
        if self.testMode == None:
            return 0
        return len(self.testcandIds)

    def takeOutputs(self, ids):

        subIds = []
        curIds = ids
        for iB in range(1, len(self.IdsBorder)):
            sIds = curIds[np.reshape(np.where(curIds < self.IdsBorder[iB]), -1)]

            subIds.append(sIds - self.IdsBorder[iB - 1])
            curIds = curIds[np.reshape(np.where(curIds >= self.IdsBorder[iB]), -1)]
        subIds.append(curIds - self.IdsBorder[-1])

        imgs = []
        for iS in range(len(subIds)):
            if len(subIds[iS]) == 0:
                continue
            if len(imgs) == 0:
                imgs = self.loaders[iS].takeOutputs(subIds[iS])
            else:
                curImgs = self.loaders[iS].takeOutputs(subIds[iS])

                if type(curImgs) == type([]):
                    for i in range(len(curImgs)):
                        imgs[i] = np.append(imgs[i], curImgs[i], axis=0)
                else:
                    imgs = np.append(imgs, curImgs, axis=0)
        return imgs

    def takeInputs(self, ids):

        subIds = []
        curIds = ids
        for iB in range(1, len(self.IdsBorder)):
            sIds = curIds[np.reshape(np.where(curIds < self.IdsBorder[iB]), -1)]

            subIds.append(sIds - self.IdsBorder[iB - 1])
            curIds = curIds[np.reshape(np.where(curIds >= self.IdsBorder[iB]), -1)]
        subIds.append(curIds - self.IdsBorder[-1])


        imgs = []
        for iS in range(len(subIds)):
            if len(subIds[iS]) == 0:
                continue
            if len(imgs) == 0:
                imgs = self.loaders[iS].takeInputs(subIds[iS])
            else:
                curImgs = self.loaders[iS].takeInputs(subIds[iS])

                if type(curImgs) == type([]):
                    for i in range(len(curImgs)):
                        imgs[i] = np.append(imgs[i], curImgs[i], axis=0)
                else:
                    imgs = np.append(imgs, curImgs, axis=0)
        return imgs

    def __updateCands(self, bRan=True, subEpochNum = 1):
        subLoaderIds = []
        for iB in range(len(self.IdsBorder)):
            subIds = self.loaders[iB].createSubEpochIds(bRan, subEpochNum).reshape((subEpochNum,-1))
            subLoaderIds.append(subIds + self.IdsBorder[iB])

        candIds = []
        subIds = []
        for iS in range(subEpochNum):

            for il in range(len(self.IdsBorder)):

                if il ==0:
                    subIds = subLoaderIds[il][iS]
                else:
                    subIds = np.append(subIds, subLoaderIds[il][iS])

            if bRan:
                idRange = np.random.permutation(len(subIds))
            else:
                idRange = np.arange(len(subIds))

            candIds = np.append(candIds, subIds[idRange]).astype(int)

        # sortids = np.sort(candIds)
        # te = np.linalg.norm(sortids - range(self.total))
        # print te

        return candIds




    def update_next_batch_Ids(self, batch_size, bRan=True, subEpochNum = 1):
        start = self.cur_id
        self.cur_id += batch_size
        if self.cur_id > self.total:
            # Finished epoch

            # Shuffle the data

            if subEpochNum == 1:
                self.candIds = np.arange(self.total)
                perm = np.arange(self.total)
                if bRan:
                    np.random.shuffle(perm)
                self.candIds = self.candIds[perm]
            else:
                self.candIds = self.__updateCands(bRan, subEpochNum)
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


    def next_inOut_batch_fromIds(self):
        outputs = self.takeOutputs(self.curIds)
        inputs = self.takeInputs(self.curIds)
        return inputs, outputs

    def setTest(self, testMode = "TestALL", **kwargs):
        self.testMode = testMode
        if self.testMode == "TestALL":
            self.testcandIds = np.arange(self.total)
        self.testcur_id = len(self.testcandIds) + 1
        print "TestSize:", self.testSize()
