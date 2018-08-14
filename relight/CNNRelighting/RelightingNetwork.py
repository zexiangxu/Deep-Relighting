from BaseNN import *

class RelightingBase(BaseNN):

    def __init__(self, featureSize= [64, 128, 256, 512], expandSize = [8, 32, 128, 128], imgSize = (128, 128), inputNum = 5):
        self.expandSize = expandSize
        self.featureSize = featureSize
        self.imgSize = imgSize
        self.inputNum = inputNum

    def _relightNet(self, inputs):

        relightInput, relightDirs, isTraining = inputs
        curLayer = relightInput

        layers = []
        layers.append(curLayer)

        # build downsampling encoder by CNN
        print "build downsample"
        for i, feature in enumerate(self.featureSize):
            with tf.variable_scope("downblock%d" % i):
                curLayerOut = self._buildConvBnRelu(curLayer, feature, isTraining, strides=2)
                layers.append(curLayerOut)
                curLayer = curLayerOut
                print curLayer
        innerLayer = layers.pop()

        # encode the relight direction by FCN
        print "expanding dirs"
        with tf.variable_scope("expandDirs"):
            curDir = tf.reshape(relightDirs, [-1, 2])
            for ii, feature in enumerate(self.expandSize):

                if ii == 0:
                    preF = 2
                else:
                    preF = self.expandSize[ii - 1]
                with tf.variable_scope("FNN%d" % ii):
                    FCNW = tf.Variable(tf.truncated_normal([preF, feature], stddev=1.0 / np.sqrt(preF)), name='weights')
                    FCNB = tf.Variable(tf.zeros([feature]), name='biases')
                    curOutDir = tf.tanh(tf.matmul(curDir, FCNW) + FCNB)
                    curDir = curOutDir
            curDir = tf.reshape(curOutDir, [-1, 1, 1, self.expandSize[-1]])
            print curDir
        centerDir = curDir * tf.constant(1.0, tf.float32, (
            1, self.imgSize[0] / 2 ** len(self.featureSize), self.imgSize[0] / 2 ** len(self.featureSize), 1))

        decoderIn = tf.concat([innerLayer, centerDir], 3)

        print "build center conv"
        with tf.variable_scope("convblock"):
            conv = self._buildConvBnRelu(decoderIn, self.featureSize[-1], isTraining)
            curLayer = tf.concat([conv, decoderIn], 3)
            print curLayer
        layers.reverse()
        invFeature = []
        for i in range(0, len(self.featureSize)):
            invFeature.append(self.featureSize[i])
        invFeature.reverse()

        # build the decoder by CNN
        print "build upsample"
        for i, feature in enumerate(invFeature):
            with tf.variable_scope("upblock%d" % i):
                curLayerOut = self._buildDeconvBnReluCont(curLayer, layers[i], feature, isTraining)
                curLayer = curLayerOut
                print curLayer

        with tf.variable_scope("out"):
            curLayer = self._buildConvBnRelu(curLayerOut, self.featureSize[0], isTraining)
            output = self._buildConvSigmoid(curLayer, 3)

        return output



class RelightingJoint(RelightingBase):
    def __init__(self, featureSize= [64, 128, 256, 512], expandSize = [8, 32, 128, 128], imgSize = (128, 128), inputNum = 4, candNum = 541, sfaB = 5.0, sfaC=0.0):
        super(RelightingJoint, self).__init__(featureSize, expandSize, imgSize, inputNum)

        self.candNum = candNum



        self.sfA = 1.0
        self.sfaB = sfaB
        self.sfaC = sfaC

    def _initialize(self):
        #the data contains all candidate input direction images
        inputCands = tf.placeholder(tf.uint8, shape=[None, self.candNum,self.imgSize[0]*self.imgSize[1]*3])
        #the data contains all candidate input direction
        input_candDirs = tf.placeholder(tf.float32, shape=[1, self.candNum, 2])

        #relighting directions for output images
        input_outDirs = tf.placeholder(tf.float32, shape=[None, 1, 1, 2])
        #tells the corresponding input of the full batch(for effectively use input data)
        input_batchSelect = tf.placeholder(tf.float32, shape=[None, None])
        #a holder contains all one, for construct the cand batch's weights
        input_WeightsHolder = tf.placeholder(tf.float32, shape=[None, 1, 1])
        #a holder contains all one, for construct all input dirs
        input_dirsHolder = tf.placeholder(tf.float32, shape=[None, 1, 1, 1])
        #a parameter to make softmax max.
        input_sfA = tf.placeholder(tf.float32, shape=1)

        #output images
        targets = tf.placeholder(tf.uint8, shape=[None, self.imgSize[0], self.imgSize[1], 3])
        #whether it is training
        isTraining = tf.placeholder(tf.bool)

        return [inputCands, input_candDirs, input_outDirs, input_batchSelect, input_WeightsHolder, input_dirsHolder, input_sfA, isTraining, targets]

    def _loss(self, graphInputs, graphOutputs):

        [inputCands, input_candDirs, input_outDirs, input_batchSelect, input_WeightsHolder, input_dirsHolder, input_sfA,
         isTraining, rawtargets] = graphInputs
        output, softMaxWeights = graphOutputs

        targets = tf.cast(rawtargets, tf.float32) / 255.0

        loss =tf.reduce_mean(tf.nn.l2_loss(output - targets))
        return loss


    def _PostTrainBatch(self, batchId, batchSize, epochSize, outFolder, graphInputs, graphOutputs):
        batchPerEp = epochSize/batchSize
        epochId = float(batchId)/(batchPerEp)
        nextEpochId = (batchId + batchPerEp - 1)/batchPerEp


        self.sfA = 1.0 + (self.sfaB * (epochId+self.sfaC))**2.0
        target = 1.0 + (self.sfaB * (nextEpochId+self.sfaC)) ** 2.0
        print "temperature: ", self.sfA, target

    def _PostTrainInfo(self, batchId, batchSize, epochSize, outFolder, outputs):
        if not os.path.isdir(outFolder+"/softWeights"):
            os.makedirs(outFolder+"/softWeights")
        imgs, softMaxWeights = outputs
        np.savetxt(outFolder+"/softWeights/%d.txt"%batchId, softMaxWeights.reshape(self.inputNum, self.candNum))


    def _makeTestInputDict(self, data_inputs, data_outputs, graphInputs, graphOutputs, **testInputArgs):
        [inputCands, input_candDirs, input_outDirs, input_batchSelect, input_WeightsHolder, input_dirsHolder, input_sfA, isTraining, targets] = graphInputs
        data_cands, data_candirs, data_outdirs, data_batchSelect, data_weightsHolder, data_inDirsHolder = data_inputs

        return {inputCands: data_cands, input_candDirs: data_candirs,
                input_outDirs: data_outdirs, input_batchSelect: data_batchSelect, input_WeightsHolder: data_weightsHolder, input_dirsHolder: data_inDirsHolder,
                input_sfA: [self.sfA],
                targets: data_outputs, isTraining: False}

    def _makeComputeInputDict(self, data_inputs, graphInputs, graphOutputs, **computeInputArgs):
        pass


    def _makeTrainInputDict(self, data_inputs, data_outputs, graphInputs, graphOutputs, **trainInputArgs):
        [inputCands, input_candDirs, input_outDirs, input_batchSelect, input_WeightsHolder, input_dirsHolder, input_sfA,
         isTraining, targets] = graphInputs
        data_cands, data_candirs, data_outdirs, data_batchSelect, data_weightsHolder, data_inDirsHolder = data_inputs

        return {inputCands: data_cands, input_candDirs: data_candirs,
                input_outDirs: data_outdirs, input_batchSelect: data_batchSelect,
                input_WeightsHolder: data_weightsHolder, input_dirsHolder: data_inDirsHolder,
                input_sfA: [self.sfA],
                targets: data_outputs, isTraining: True}

    # build the whole graph
    def _BuildGraph(self, inputs):
        # depacketize the input
        rawCands, input_candDirs, input_outDirs, input_batchSelect, input_WeightsHolder, input_dirsHolder, input_sfA, isTraining, _ = inputs

        # cast the input into float32
        inputCands = tf.cast(rawCands, tf.float32) / 255.0 * 2.0 - 1.0

        # Sample-Net
        with tf.variable_scope("selecting"):
            relightInput, input_outDirs, softMaxWeights = self._sampleNet([inputCands, input_candDirs, input_outDirs, input_batchSelect, input_WeightsHolder, input_dirsHolder, input_sfA, isTraining, _])

        # Relight-Net
        output = self._relightNet([relightInput, input_outDirs, isTraining])

        return [output, softMaxWeights]


    def _sampleNet(self, inputs):
        inputCands, input_candDirs, input_outDirs, input_batchSelect, input_WeightsHolder, input_dirsHolder, input_tmp, isTraining, _ = inputs

        # construct optWeights, initialize with all 1.0, average
        optWeights = tf.Variable(tf.constant(1.0, tf.float32, (1, self.inputNum, self.candNum)), name="optWeights")

        # times the temperature parameter
        multOptWeights = input_tmp * optWeights

        # through softmax, achieve the linear weights to construct input
        softMaxWeights = tf.nn.softmax(multOptWeights, 2, name="softWeights")

        # construct (expand) the batch weights for all input cands
        batchWeights = input_WeightsHolder  *  softMaxWeights

        # linear combine input candidates by softmax weights
        selectCands = tf.matmul(batchWeights, inputCands, name="selectCands")

        # construct the full batch input images for the Relight-Net by doing matrix multiplication
        selectCandsData = tf.reshape(selectCands, (-1, self.inputNum * self.imgSize[0] * self.imgSize[1] * 3))
        fullbatchInputs = tf.matmul(input_batchSelect, selectCandsData)

        # reshape and swap to a batch shape
        fullbatchCandImgs = tf.reshape(fullbatchInputs, (-1, self.inputNum, self.imgSize[0], self.imgSize[1], 3))
        fullbatchCandsChanel2 = tf.transpose(fullbatchCandImgs, [0, 2, 3, 1, 4])
        fullbatchCandsChanel = tf.reshape(fullbatchCandsChanel2,
                                          (-1, self.imgSize[0], self.imgSize[1], self.inputNum * 3))

        # linear combine the directions
        selectDirs = tf.matmul(softMaxWeights, input_candDirs)
        selectDirsData = tf.reshape(selectDirs, (1, 1, 1, self.inputNum * 2))

        # construct the full batch directions
        fullbatchDirs = input_dirsHolder * tf.constant(1.0, tf.float32,  [1, self.imgSize[0], self.imgSize[1], 1]) * selectDirsData

        # construct the input for relight net
        relightInput = tf.concat([fullbatchCandsChanel, fullbatchDirs], 3, "concatRelight")

        return relightInput, input_outDirs, softMaxWeights


class RelightingNet(RelightingBase):
    def __init__(self, featureSize=[64, 128, 256, 512], expandSize=[8, 32, 128, 128], imgSize=(256, 256), inputNum=4):
        super(RelightingNet, self).__init__(featureSize, expandSize, imgSize, inputNum)



    def _initialize(self):

        inputNum = self.inputNum
        inputImgs = tf.placeholder(tf.uint8, shape=[None, self.imgSize[0], self.imgSize[1], inputNum * 3])
        inputDirs = tf.placeholder(tf.float32, shape=[None, 1, 1, inputNum * 2])
        relightDirs = tf.placeholder(tf.float32, shape=[None, 1, 1, 2])
        targets = tf.placeholder(tf.uint8, shape=[None, self.imgSize[0], self.imgSize[1], 3])
        isTraining = tf.placeholder(tf.bool)

        return [inputImgs, inputDirs, relightDirs, isTraining, targets]

    def _makeTestInputDict(self, data_inputs, data_outputs, graphInputs, graphOutputs, **testInputArgs):
        inputImgs, inputDirs, relightDirs, isTraining, targets = graphInputs
        data_Imgs, data_inDirs, data_outDirs = data_inputs

        return {inputImgs: data_Imgs, inputDirs: data_inDirs, relightDirs: data_outDirs, targets: data_outputs,
                isTraining: False}

    def _makeComputeInputDict(self, data_inputs, graphInputs, graphOutputs, **computeInputArgs):
        inputImgs, inputDirs, relightDirs, isTraining, targets = graphInputs
        data_Imgs, data_inDirs, data_outDirs = data_inputs

        return {inputImgs: data_Imgs, inputDirs: data_inDirs, relightDirs: data_outDirs, isTraining: False}

    def _makeTrainInputDict(self, data_inputs, data_outputs, graphInputs, graphOutputs, **trainInputArgs):
        inputImgs, inputDirs, relightDirs, isTraining, targets = graphInputs
        data_Imgs, data_inDirs, data_outDirs = data_inputs

        return {inputImgs: data_Imgs, inputDirs: data_inDirs, relightDirs: data_outDirs, targets: data_outputs,
                isTraining: True}

    def _loss(self, graphInputs, graphOutputs):

        inputImgs, inputDirs, relightDirs, isTraining, targets = graphInputs

        out = tf.cast(targets, tf.float32) / 255.0
        loss = tf.reduce_mean(tf.nn.l2_loss(graphOutputs - out))
        return loss

    def _BuildGraph(self, inputs):
        inputImgs, inputDirs, relightDirs, isTraining, _ = inputs

        inImgs = tf.cast(inputImgs, tf.float32) / 255.0 * 2.0 - 1.0
        inDirs = inputDirs * tf.constant(1.0, tf.float32, (1, self.imgSize[0], self.imgSize[1], 2 * self.inputNum))
        relightIn = tf.concat([inImgs, inDirs], axis=3)

        output = self._relightNet([relightIn, relightDirs, isTraining])

        return output





