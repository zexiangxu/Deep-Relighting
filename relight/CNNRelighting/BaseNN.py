import tensorflow as tf
import numpy as np
import os


class BaseNN(object):

    #building the graph
    def _initialize(self):
        pass

    def _BuildGraph(self, inputs):
        pass

    def _makeTrainInputDict(self, data_inputs, data_outputs, graphInputs, graphOutputs, **trainInputArgs):
        pass
    def _makeDebugInputDict(self, data_inputs, data_outputs, graphInputs, graphOutputs, **trainInputArgs):
        pass

    def _makeTestInputDict(self, data_inputs, data_outputs, graphInputs, graphOutputs, **testInputArgs):
        pass

    def _makeComputeInputDict(self, data_inputs, graphInputs, graphOutputs, **computeInputArgs):
        pass

    def _loss(self, graphInputs, graphOutputs):
        pass

    def _multiLosses(self, graphInputs, graphOutputs):
        pass

    def _makeTrainSteps(self, tf_learning_rate, losses):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_steps = []
        with tf.control_dependencies(update_ops):

            for il,loss in enumerate(losses):
                one_step = tf.train.AdamOptimizer(tf_learning_rate).minimize(loss)
                train_steps.append(one_step)
        return train_steps

    def _buildBnRelu(self, inputs, isTraining, axis=3):
        bn1 = tf.layers.batch_normalization(inputs, axis=axis, training=isTraining, name="bn1")
        relu1 = tf.nn.relu(bn1, name="relu1")
        return relu1

    def _buildConvBnRelu2(self, inputs, filters,  isTraining, strides = [1, 1], axis=3):
        conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=[3, 3],
            strides=strides[0],
            padding="same",
            name="conv1")
        bn1 = tf.layers.batch_normalization(conv1, axis=axis, training=isTraining, name="bn1")
        relu1 = tf.nn.relu(bn1, name="relu1")
        conv2 = tf.layers.conv2d(
            inputs=relu1,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=[3, 3],
            strides=strides[1],
            padding="same",
            activation=tf.nn.relu, name="conv2")
        bn2 = tf.layers.batch_normalization(conv2, axis=3, training=isTraining, name="bn2")
        relu2 = tf.nn.relu(bn2, name="relu2")
        return relu2

    def _buildConvBnRelu(self, inputs, filters,  isTraining, strides = 1, kernel_size = [3,3], axis=3):
        conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            name="conv")
        bn1 = tf.layers.batch_normalization(conv1, axis=axis, training=isTraining, name="bn")
        relu1 = tf.nn.relu(bn1, name="relu")
        return relu1



    def _buildBnReluConv(self, inputs, filters,  isTraining, strides = 1, kernel_size=[3, 3], axis=3):
        bn1 = tf.layers.batch_normalization(inputs, axis=axis, training=isTraining, name="bn")
        relu1 = tf.nn.relu(bn1, name="relu")
        conv1 = tf.layers.conv2d(
            inputs=relu1,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            name="conv1")
        return conv1

    def _buildDenseBlock(self, previousInputs, filters, isTraining, blockSize = 5):
        curPrevious = previousInputs
        for i in range(blockSize):
            with tf.variable_scope("dense_%d"%i):
                cont = tf.concat(curPrevious, 3)
                with tf.variable_scope("Bottleneck"):
                    conv1 = self._buildBnReluConv(cont, filters*4, isTraining, kernel_size=[1,1])
                with tf.variable_scope("Conv33"):
                    conv2 = self._buildBnReluConv(conv1, filters, isTraining, kernel_size=[3,3])
                curPrevious.append(conv2)
        out = tf.concat(curPrevious, 3)
        return out

    def _buildConvRelu(self, inputs, filters):
        conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=[3, 3],
            padding="same",
            name="conv1")
        relu1 = tf.nn.relu(conv1, name="relu1")

        return relu1

    def _buildConvSigmoid(self, inputs, filters):
        conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=[3, 3],
            padding="same",
            name="conv1")
        sig1 = tf.nn.sigmoid(conv1, name="sig")

        return sig1

    def _buildConvTanh(self, inputs, filters):
        conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=[3, 3],
            padding="same",
            name="conv1")
        tanh1 = tf.nn.tanh(conv1, name="tanh")

        return tanh1


    def _buildConvReluPoolBlock(self, inputs, filters, isTraining):
        conv = self._buildConvBnRelu2(inputs, filters, isTraining)
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2, name="pool")

        return pool

    def _buildUpsampleConvReluBlock(self, inputs, downsamplelayer, filters, isTraining):
        upsample = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=[3, 3],
            strides=2,
            padding="same",
            activation=tf.nn.relu, name="deconv")
        cont = tf.concat([upsample, downsamplelayer], 3)
        print upsample, downsamplelayer, cont
        conv = self._buildConvBnRelu2(cont, filters, isTraining)
        return conv

    def _buildDeconvBnReluCont(self, inputs, contlayer, filters, isTraining, axis=3):
        upsample = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=[3, 3],
            strides=2,
            padding="same",
            name="deconv")
        bn1 = tf.layers.batch_normalization(upsample, axis=axis, training=isTraining, name="bn")
        relu1 = tf.nn.relu(bn1, name="relu")
        if contlayer == None:
            return relu1
        cont = tf.concat([relu1, contlayer], 3)
        return cont

    def _buildBottleneckDeconvBnReluCont(self, inputs, contlayer, filters, isTraining, axis=3):
        with tf.variable_scope("bottleneck"):
            weighted = self._buildConvBnRelu(inputs, filters, isTraining, strides=1, kernel_size=[1,1], axis=axis)

        upsample = tf.layers.conv2d_transpose(
            inputs=weighted,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=[3, 3],
            strides=2,
            padding="same",
            name="deconv")
        bn1 = tf.layers.batch_normalization(upsample, axis=axis, training=isTraining, name="bn")
        relu1 = tf.nn.relu(bn1, name="relu")
        if contlayer == None:
            return relu1
        cont = tf.concat([relu1, contlayer], 3)
        return cont



    def _testInTrain(self, outFolder, tf_sess, tf_loss, tf_inputs, tf_outputs, testSet, batchSize=10, maxSize=1000, bSave=True, saveFunc = None,
                     **kwargs
                      ):
        maxSize = min(maxSize, testSet.testSize())
        batchNum = (maxSize + batchSize - 1) / batchSize
        print "test all: %d" % (batchNum)
        totalLoss = 0
        for i in range(batchNum):
            if batchNum / 100 == 0 or i % (batchNum / 100) == 0 or i == batchNum - 1:
                print i,
            data_inputs, data_outputs = testSet.next_test_batch(batchSize)
            te_loss, outputs = tf_sess.run([tf_loss, tf_outputs],
                    feed_dict=self._makeTestInputDict(data_inputs, data_outputs, tf_inputs, tf_outputs,**kwargs))
            totalLoss += te_loss
            if bSave:
                if saveFunc == None:
                    testSet.saveCurOutputsSameFolder(outputs, outFolder, **kwargs)
                else:
                    saveFunc(outputs, outFolder, **kwargs)
        print "test complete"
        return totalLoss / maxSize


    def Compute(self, outFolder, dataSet, restorePath, batchSize = 1, saveFunc = None, **kwargs):
        tf.reset_default_graph()

        graphInputs = self._initialize()
        graphOutputs = self._BuildGraph(graphInputs)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        if restorePath != None and os.path.isfile(restorePath + '.index'):
            saver.restore(sess, restorePath)
            print "weights restored"
        else:
            print "no weights!!!!!!!!!!!!!!!!!!!!!!!!"
            return
        numBatch = (dataSet.testSize() + batchSize - 1) / batchSize

        print "test all: %d" % (numBatch)

        for i in range(numBatch):
            print i, (numBatch)
            data_inputs = dataSet.next_compute_batch(batchSize, **kwargs)
            outputs = sess.run(graphOutputs, feed_dict=
            self._makeComputeInputDict(data_inputs, graphInputs, graphOutputs, **kwargs))

            if saveFunc == None:
                dataSet.saveCurOutputs(outputs, outFolder, **kwargs)
            else:
                saveFunc(outputs, outFolder, **kwargs)
        print "compute complete"




    def Test(self, outFolder, testSet, restorePath, batchSize = 15, bSave = True, **kwargs):


        tf.reset_default_graph()

        graphInputs = self._initialize()
        graphOutputs = self._BuildGraph(graphInputs)

        loss = self._loss(graphInputs, graphOutputs)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        if restorePath != None and os.path.isfile(restorePath+'.index'):
            saver.restore(sess, restorePath)
            print "weights restored"
        else:
            print "no weights!!!!!!!!!!!!!!!!!!!!!!!!"
            return
        numBatch = (testSet.testSize() + batchSize -1) / batchSize

        print "test all: %d" % (numBatch)
        totalLoss = None
        losses = []

        for i in range(numBatch):
            print i, (numBatch)
            data_inputs, data_outputs = testSet.next_test_batch(batchSize)
            te_loss, outputs = sess.run([loss, graphOutputs], feed_dict =
                self._makeTestInputDict(data_inputs,data_outputs, graphInputs, graphOutputs, **kwargs))
            if type(totalLoss) == type(None):
                totalLoss = np.reshape(te_loss, -1)
                losses = np.reshape(te_loss, (1,-1))
            else:
                totalLoss += te_loss
                losses = np.append(losses, np.reshape(te_loss, (1,-1)), axis=0)

            # outImg = (np.clip(outputs[0]* 255.0 + 0.5, 0, 255) ).astype(np.uint8)

            print totalLoss / (i*batchSize + batchSize)
            if bSave:
                testSet.saveCurOutputs(outputs, outFolder, **kwargs)
        print "test complete"

        np.savetxt(outFolder + "/losses.txt", losses[:,0])
        np.savetxt(outFolder + "/alllosses.txt", losses)

        for il in range(1, losses.shape[1]):
            oneloss = losses[:,il]
            np.savetxt(outFolder + "/losses_%d.txt"%il, oneloss)
        print totalLoss / testSet.testSize()
        return totalLoss / testSet.testSize()


    # do something after training a batch
    def _PostTrainBatch(self, batchId, batchSize, epochSize, outFolder, graphInputs, graphOutputs):
        pass
    # do something before training a batch
    def _PreTrainBatch(self, batchId, batchSize, epochSize, outFolder):
        pass
    # do somthing after training an sub epoch
    def _PostTrainInfo(self, batchId, batchSize, epochSize, outFolder, outputs):
        pass


    def TrainTest_continue(self, outFolder, trainingSet, testSet, learningRate, batchSize=12, testBatch = 10, savePerEpoch=2, trainInfoPerEpoch=100,
                  nEpochs=20, restorePath = None, testMaxSize = 100, bSaveTest = True, saveTestFunc = None , **kwargs):
        # reset the graph
        tf.reset_default_graph()

        # initialize the input and output variables of the graph
        graphInputs = self._initialize()
        # build the graph
        graphOutputs = self._BuildGraph(graphInputs)

        # make the loss function, prepare the learning rate variable
        loss = self._loss(graphInputs, graphOutputs)
        tf_learning_rate_train = tf.placeholder(tf.float32)

        # for the batch normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(tf_learning_rate_train).minimize(loss)

        # conservative memory strategy
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        #create the session and saver
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=1000)


        # compute the epochSize, saveSize, and the size to check an average loss
        epochSize = trainingSet.epochSize(batchSize = batchSize, **kwargs)

        saveSize = epochSize / savePerEpoch / batchSize
        trainInfoSize = epochSize / trainInfoPerEpoch / batchSize
        trLoss = []
        teLoss = []
        epochTrLoss = []

        # create the folder for storing the weights
        weightsFolder = outFolder + "/weights"
        if not os.path.isdir(weightsFolder):
            os.makedirs(weightsFolder)

        # print the sizes
        print "epochSize: %d, batchSize: %d, trainInfoSize %d, saveInfo size %d" % (
        epochSize, batchSize, trainInfoSize, saveSize)

        # load the latest trained weight of an epoch if there is one.
        # this assumes the latest training used the same (epoch size) parameters
        weightId = 0
        for i in range(2, nEpochs * savePerEpoch,2):
            if os.path.isfile(weightsFolder + "/w_%d.ckpt.index" % (i)):
                weightId = i
            else:
                break


        # load previous trained weights if possible
        if weightId != 0:
            print weightId
            saver.restore(sess, weightsFolder + "/w_%d.ckpt"%weightId)
            print "weights restored %d"%weightId

            if os.path.isfile(outFolder + "/epochtrLoss.txt"):
                orgEpochtrLoss = np.loadtxt(outFolder + "/epochtrLoss.txt")
            else:
                orgEpochtrLoss = []
            if os.path.isfile(outFolder + "/trLoss.txt"):
                orgtrLoss = np.loadtxt(outFolder + "/trLoss.txt")
            else:
                orgtrLoss = []
            if os.path.isfile(outFolder + "/teLoss.txt"):
                orgteLoss = np.loadtxt(outFolder + "/teLoss.txt")
            else:
                orgteLoss = []

            id = saveSize * weightId / trainInfoSize
            for ii in orgtrLoss[:id]:
                trLoss.append(ii)
            for ii in orgEpochtrLoss[:weightId]:
                epochTrLoss.append([ii])
            for ii in orgteLoss[:weightId]:
                teLoss.append(ii)


            self._PostTrainBatch(saveSize * weightId, batchSize, epochSize, outFolder, graphInputs, graphOutputs)

        # load weights from a specified path
        elif restorePath != None and os.path.isfile(restorePath + ".index"):
            saver.restore(sess, restorePath)
            print "weights restored from path"
        else:
            print "start from beginning"


        epochError = 0
        trainInfoError = 0
        epochNum = 0
        trainInfoNum = 0
        for i in range(saveSize * weightId + 1, (epochSize + batchSize - 1) / batchSize * nEpochs):


            self._PreTrainBatch(i, batchSize, epochSize, outFolder)

            print "cur batch: %d, next saveInfo: %d, next epoch: %d"%\
                  (i, ((i + 1) / trainInfoSize + 1) * trainInfoSize, ((i + 1) / saveSize + 1) * saveSize)
            data_inputs, data_outputs = trainingSet.next_batch(batchSize, **kwargs)

            # create the input dictionary
            trainDict = {tf_learning_rate_train: learningRate}
            trainDict.update(self._makeTrainInputDict(data_inputs, data_outputs, graphInputs, graphOutputs, **kwargs))
            _, tr_loss = sess.run([train_step, loss], feed_dict=trainDict)


            epochError += tr_loss / batchSize
            trainInfoError += tr_loss / batchSize
            epochNum += 1
            trainInfoNum += 1

            print "epoch: %d, %d, epochLoss: %f, checkLoss: %f, loss: %f" % (
                (i + 1) * batchSize / epochSize, (i + 1) / trainInfoSize,
                epochError / epochNum,
                trainInfoError / trainInfoNum,
                tr_loss / batchSize)

            # save the average loss of a sub epoch (trainInfoSize) of data
            if (i + 1) % trainInfoSize == 0:
                print "\ntest train point"

                trLoss.append([epochError / epochNum, trainInfoError / trainInfoNum, tr_loss / batchSize])
                trainInfoError = 0
                trainInfoNum = 0

                np.savetxt(outFolder + "/trLoss.txt", np.reshape(trLoss, (-1, 3)))

                info_outputs = sess.run(graphOutputs, feed_dict=trainDict)
                self._PostTrainInfo(i, batchSize, epochSize, outFolder, info_outputs)

            # save the weights and loss and test the model on a validate dataset.
            if (i + 1) % saveSize == 0:
                saver.save(sess, weightsFolder + "/w_%d.ckpt" % ((i + 1) / saveSize))

                print "\nsave point"
                epochTrLoss.append([epochError / epochNum])
                epochError = 0
                epochNum = 0
                np.savetxt(outFolder + "/epochtrLoss.txt", np.reshape(epochTrLoss, (-1)))

                te_loss = self._testInTrain(outFolder + "/testOut/%d" % ((i + 1) / saveSize), sess,
                                             loss, graphInputs, graphOutputs, testSet, batchSize=testBatch, maxSize=testMaxSize,
                                             bSave=bSaveTest, saveFunc=saveTestFunc, **kwargs)

                print "test loss: %f" % (
                    te_loss)
                teLoss.append(te_loss)
                np.savetxt(outFolder + "/teLoss.txt", np.reshape(teLoss, -1))


            self._PostTrainBatch(i, batchSize, epochSize, outFolder, graphInputs, graphOutputs)


