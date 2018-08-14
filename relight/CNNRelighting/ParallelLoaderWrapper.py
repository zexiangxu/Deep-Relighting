
import threading
from collections import deque
import copy



class LoadingThread(threading.Thread):
    def __init__(self, loader, func, locker, dataVec):
        super(LoadingThread, self).__init__()
        self.func = func
        self.loader = loader

        self.locker = locker
        self.dataVec = dataVec
    def run(self):

        data = self.func()
        self.locker.acquire()
        self.dataVec.append((self.loader, data))
        self.locker.release()

class SavingThread(threading.Thread):
    def __init__(self, func, outputs, outFolder, params):
        super(SavingThread, self).__init__()
        self.func = func

        self.params = params
        self.outputs = outputs
        self.outFolder = outFolder

    def run(self):
        if type(self.params) == type({}):
            self.func(self.outputs, self.outFolder, **self.params)
        else:
            self.func(self.outputs, self.outFolder)

#A simple wrapper that can do parallel loading batches. The order of batches may change.
#
class ParallelLoaderWrapper(object):
    def __init__(self, dataLoader):
        self.dataLoader = dataLoader
        self.locker = threading.Lock()
        self.trainDataVec = deque([])
        self.testDataVec = deque([])
        self.computeDataVec = deque([])
        self.currentST = None
        self.currentTT = None
        self.currentTeT = None
        self.currentCT = None
    def next_compute_batch(self, batch_size, **kwargs):

        if self.currentCT == None:
            self.dataLoader.update_next_test_batch_Ids(batch_size, **kwargs)
            curLoader = copy.copy(self.dataLoader)
            self.currentCT = LoadingThread(curLoader, curLoader.next_in_batch_fromIds, self.locker, self.computeDataVec)
            self.currentCT.start()
        self.dataLoader.update_next_test_batch_Ids(batch_size, **kwargs)
        curLoader = copy.copy(self.dataLoader)
        CT = LoadingThread(curLoader, curLoader.next_in_batch_fromIds, self.locker, self.computeDataVec)
        CT.start()
        self.currentCT.join()
        self.currentCT = CT
        self.locker.acquire()
        curloader, data = self.computeDataVec.popleft()
        self.curloader = curloader
        self.locker.release()

        return data

    def epochSize(self, **kwargs):
        return self.dataLoader.epochSize(**kwargs)

    def testSize(self):
        return self.dataLoader.testSize()


    def next_batch(self, batch_size, **kwargs):

        if self.currentTT == None:
            self.dataLoader.update_next_batch_Ids(batch_size, **kwargs)
            curLoader = copy.copy(self.dataLoader)
            self.currentTT = LoadingThread(curLoader, curLoader.next_inOut_batch_fromIds, self.locker, self.trainDataVec)
            self.currentTT.start()
        self.dataLoader.update_next_batch_Ids(batch_size, **kwargs)
        curLoader = copy.copy(self.dataLoader)
        TT = LoadingThread(curLoader, curLoader.next_inOut_batch_fromIds, self.locker, self.trainDataVec)
        TT.start()
        self.currentTT.join()
        self.currentTT = TT
        self.locker.acquire()
        curloader, data = self.trainDataVec.popleft()
        self.curloader = curloader
        self.locker.release()

        return data


    def next_test_batch(self, batch_size):
        if self.currentTeT == None:
            self.dataLoader.update_next_test_batch_Ids(batch_size)
            curLoader = copy.copy(self.dataLoader)
            self.currentTeT = LoadingThread(curLoader, curLoader.next_inOut_batch_fromIds, self.locker, self.testDataVec)
            self.currentTeT.start()
        self.dataLoader.update_next_test_batch_Ids(batch_size)
        curLoader = copy.copy(self.dataLoader)
        TeT = LoadingThread(curLoader, curLoader.next_inOut_batch_fromIds, self.locker, self.testDataVec)
        TeT.start()
        self.currentTeT.join()
        self.currentTeT = TeT
        self.locker.acquire()
        curloader, data = self.testDataVec.popleft()
        self.curloader = curloader
        self.locker.release()
        return data


    def saveCurOutputs(self, outputs, outFolder, **kwargs):
        if self.currentST != None:
            self.currentST.join()
        self.currentST = SavingThread(self.curloader.saveCurOutputs, outputs, outFolder, kwargs)
        self.currentST.start()



    def saveCurOutputsSameFolder(self, outputs, outFolder, **kwargs):
        if self.currentST != None:
            self.currentST.join()
        self.currentST = SavingThread(self.curloader.saveCurOutputsSameFolder, outputs, outFolder, kwargs)
        self.currentST.start()
