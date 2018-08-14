from PIL import ImageDraw

from ..CNNRelighting.ComputeLoader import *
from ..CNNRelighting.ParallelLoaderWrapper import *
from ..Common import *
import cv2
import tensorflow as tf

#
#class Animator to make video under different lighting conditions
#it starts by using CNN relighter to create all necessary images under different directional lights, called cands(candidates)
#it also creates per frame weights which are used to combine those cands
class Animator(object):
    def __init__(self, relightor, inImgs, inDirs, candFolder=None):

        if relightor != None:
            self.relightor = relightor

        if inDirs != None:
            self.inDirs = inDirs
        if inImgs != None and inDirs != None:
            self.inImgs = inImgs
            if relightor.__class__.__name__ == "RelightingVaryingInput":
                orgLoader = ComputeLoader(inImgs, inDirs)
            else:
                orgLoader = ComputeLoader_sep(inImgs, inDirs)
            self.dataLoader = ParallelLoaderWrapper(orgLoader)
        self.chosenIds = None
        self.candFolder = candFolder
        if candFolder == None:
            self.candFolder = "cands"
        self.flipOpt = None
    #load candidate images
    #this is called when images are not very many
    def loadAllCands(self, imgFolder, maxSize = 99999):
        imgs = []
        for i in range(min(self.dataLoader.testSize(), maxSize)):
            img = np.power(np.asarray(Image.open(imgFolder + "/%d_%.3f_%.3f.png" % (
                i,
                self.dataLoader.dataLoader.Coefs[i][0], self.dataLoader.dataLoader.Coefs[i][1])), np.float32)/255.0, 2.2)
            imgs.append(img)

        self.imgSize = (img.shape[0], img.shape[1])
        self.imgs = np.reshape(imgs, (-1,) + self.imgSize + (3,))

    #generate all candidate directions that will be used to create all cands
    def _genCandDirs(self):
        pass

    #generate per frame weight, which will be used to combine all candidates
    def _genFrameMaps(self):
        pass

    #generate each frame's light image for demonstration
    def _genLightImgs(self, outFolder, candDirs, weightMaps):
        pass

    #check which candidates will be relighted
    def checkTargets(self, folder, coefs):
        ids = []
        if not os.path.isdir(folder):
            return range(len(coefs))

        for id in range(len(coefs)):
            if not os.path.isfile(folder+"/%d_%.3f_%.3f.png"%(id, coefs[id][0], coefs[id][1])):
                ids.append(id)

        return ids

    #make light images only, without creae frames
    def makeLightImgs(self, outFolder, outPrefix="", **kwargs):
        allCandDirs = self._genCandDirs()


        weights, bSparse = self._genFrameMaps()



        print "making light images"
        if not os.path.isdir(outFolder + "/lightImg_%s" % (outPrefix)):
            os.makedirs(outFolder + "/lightImg_%s" % (outPrefix))

        self._genLightImgs(outFolder + "/lightImg_%s" % outPrefix, allCandDirs, [weights, bSparse])

    # make candImgs
    def makeCandLightImgs(self, outFolder, inCoefs, **kwargs):
        # generate candidate dirs
        if not os.path.isdir(outFolder + "/candLs"):
            os.makedirs(outFolder + "/candLs")

        allCandDirs = self._genCandDirs()

        test = np.ones((511, 511, 3), np.uint8) * 255

        image = Image.fromarray(test)
        draw = ImageDraw.Draw(image)
        # draw.ellipse((0, 0, 510, 510), fill='black')
        borderXYL = 1.0
        draw.ellipse((255 - int(255 * borderXYL + 0.5), 255 - int(255 * borderXYL + 0.5),
                      255 + int(255 * borderXYL + 0.5), 255 + int(255 * borderXYL + 0.5)),
                     fill=(0, 0, 0))
        inCoefs = (inCoefs * 511).astype(int)
        test = np.asarray(image, np.uint8).copy()
        uwidth = 10
        for iw, dir in enumerate(inCoefs):
            test[max(0, dir[1] - uwidth):dir[1] + uwidth + 1,
            max(0, dir[0] - uwidth):dir[0] + uwidth + 1] = (128, 128, 128)
        img = test
        allCoefs = (allCandDirs[:,:2]*0.5+0.5)
        for ii, outDir in enumerate(allCoefs):
            dir = (outDir * 511).astype(int)
            test = img.copy()
            test[max(0, dir[1] - uwidth):dir[1] + uwidth + 1,
            max(0, dir[0] - uwidth):dir[0] + uwidth + 1] = (255, 255, 0)

            Image.fromarray(test).save(outFolder + "/candLs" + "/%d_%.3f_%.3f.png"%(ii, outDir[0], outDir[1]))






    #make animation
    def animate(self, outFolder, restorePath, outPrefix="", bCalcCand = True, bMakeLight = True, **kwargs):
        #generate candidate dirs
        if not os.path.isdir(outFolder + "/frames_%s" % (outPrefix)):
            os.makedirs(outFolder + "/frames_%s" % (outPrefix))

        allCandDirs = self._genCandDirs()
        if not os.path.isdir(outFolder):
            os.makedirs(outFolder)
        #set target dirs to loader, which will be feed to relighter to generate all cands
        self.dataLoader.dataLoader.setTest(relightDirs=allCandDirs)
        np.save(outFolder+"/envDirs.npy", allCandDirs)
        np.savetxt(outFolder + "/envDirs.txt", allCandDirs)
        #check which cand images are not relighted
        ids = self.checkTargets(outFolder+"/" +self.candFolder, self.dataLoader.dataLoader.Coefs)

        #relight all necessary images
        print "computing %d images"%len(ids)
        if len(ids) != 0:
            if not bCalcCand:
                print "Not Enough Cand!"
                return
            self.dataLoader.dataLoader.setTest(relightDirs=allCandDirs, selectTargets=ids)
            self.relightor.Compute(outFolder+"/"+self.candFolder, self.dataLoader, restorePath, **kwargs)


        #generate per frame weight maps, which can be sparse that contain candids and weights
        #or dense, which contain a matrix for weights of all cands
        weights, bSparse = self._genFrameMaps()


        if bMakeLight:
            print "making light images"
            if not os.path.isdir(outFolder + "/lightImg_%s" % (outPrefix)):
                os.makedirs(outFolder + "/lightImg_%s" % (outPrefix))

            #generate light images
            self._genLightImgs(outFolder + "/lightImg_%s" % outPrefix, allCandDirs, [weights, bSparse])

        #if weights are sparse or images are not too many, load all cands
        if bSparse or self.dataLoader.testSize() < 400:
            #load all and make

            print "loading all images"
            self.loadAllCands(outFolder + "/"+self.candFolder)

            print "making frames"
            for iw, weight in enumerate(weights):
                if bSparse:
                    frame = np.power(np.sum(self.imgs[weight[0]] * np.reshape(weight[1], (-1,1,1,3)), axis=0), 1.0/2.2)*255.0
                else:
                    frame = np.power(np.sum(self.imgs * weight.reshape((-1,1,1,3)), axis=0), 1.0/2.2)*255.0

                frame[frame > 255] = 255


                Image.fromarray(frame.astype(np.uint8)).save(
                    outFolder + "/frames_%s/%04d.png" % (outPrefix, iw))
        #when images are too many, create all frames at the same time by adding per cand with calculated weights
        else:
            print "making frames"
            weights = np.reshape(weights, (-1, )+weights[0].shape)
            if self.chosenIds == None:
                chosenIds = range(weights.shape[1])
            else:
                chosenIds = self.chosenIds
            print "choosing ", len(chosenIds), "from ", weights.shape[1]
            self.loadAllCands(outFolder + "/"+self.candFolder, 1)
            results = np.zeros((len(weights), ) + self.imgSize + (3,), np.float32)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            tf_outImg, [tf_targetImgs, tf_weights, tf_img] = self.gpu_adder(results)
            batch = 50

            for i in chosenIds:
                print i,
                if (i+1) % 30 == 0:
                    print " "
                # img = np.power(np.asarray(Image.open(outFolder + "/cands" + "/Shape__0/0/inters/%d_%.3f_%.3f.png" % (
                #     i,
                #     self.dataLoader.dataLoader.Coefs[i][0], self.dataLoader.dataLoader.Coefs[i][1])),
                #                           np.float32) / 255.0, 2.2)
                img = np.asarray(Image.open(outFolder + "/"+self.candFolder + "/%d_%.3f_%.3f.png" % (
                    i,
                    self.dataLoader.dataLoader.Coefs[i][0], self.dataLoader.dataLoader.Coefs[i][1])), np.uint8).reshape((1,)+ self.imgSize + (3,))
                ib = 0
                while ib<len(results):
                    results[ib:ib+batch] = sess.run(tf_outImg,
                            feed_dict={tf_targetImgs:results[ib:ib+batch], tf_weights:weights[ib:ib+batch,i,:].reshape(-1, 1, 1, 3), tf_img:img})
                    ib+=batch
                # results += img * weights[:,i,:].reshape(-1, 1, 1, 3)
            results = np.power(results, 1.0/2.2)*255.0
            results[results > 255] = 255
            for i, frame in enumerate(results):

                Image.fromarray(frame.astype(np.uint8)).save(
                    outFolder + "/frames_%s/%04d.png" % (outPrefix, i))

    def gpu_adder(self, targetImgs):
        tf_targetImgs = tf.placeholder(tf.float32, shape=[None, targetImgs[0].shape[0], targetImgs[0].shape[1], 3])
        tf_weights = tf.placeholder(tf.float32, shape=[None, 1, 1, 3])
        tf_img = tf.placeholder(tf.uint8, shape=[1, targetImgs[0].shape[0], targetImgs[0].shape[1], 3])

        inImg = tf.pow(tf.cast(tf_img, tf.float32) / 255.0, 2.2)

        outImg = tf_targetImgs + inImg * tf_weights

        return outImg, [tf_targetImgs, tf_weights, tf_img]




#Env animator, render images with a rotating environment map.
class EnvFineCircleAnimator(Animator):
    def __init__(self, relightor, inImgs, inDirs, imgRate = 0.01, nSample = 3, candFolder = None):
        self.imgRate = imgRate
        self.nSample = nSample
        super(EnvFineCircleAnimator, self).__init__(relightor, inImgs, inDirs, candFolder)

    def makeLightImgs(self, outFolder, outPrefix="", envMap = None, deg = 45, rotRes=200, lightRes=257 ):

        self.rotRes = rotRes
        self.lightRes = lightRes
        self.envMap = envMap
        self.envRes = self.envMap.shape[0]
        self.deg = deg

        super(EnvFineCircleAnimator, self).makeLightImgs(outFolder, outPrefix="env_" + outPrefix)
    def makeCandLightImgs(self, outFolder, inCoefs, envMap = None, deg = 45, rotRes=200, lightRes=257, **kwargs):
        self.rotRes = rotRes
        self.lightRes = lightRes
        self.envMap = envMap
        self.envRes = self.envMap.shape[0]
        self.deg = deg
        super(EnvFineCircleAnimator, self).makeCandLightImgs(outFolder, inCoefs, **kwargs)
    def animate(self, outFolder, restorePath, outPrefix="", envMap = None, deg = 45, rotRes=200, lightRes=257, **kwargs ):
        self.rotRes = rotRes
        self.lightRes = lightRes
        self.envMap = envMap
        self.envRes = self.envMap.shape[0]
        self.deg = deg

        #flipOpt specify whether the input images are flipped. Sometimes,the original light directions are too far from optimal, which requires flipping images with filpped directions.
        if kwargs.has_key("flipOpt") and len(kwargs["flipOpt"]) == 2:
            self.flipOpt = kwargs["flipOpt"]
        else:
            self.flipOpt = None

        super(EnvFineCircleAnimator, self).animate(outFolder, restorePath, outPrefix="env_"+outPrefix, **kwargs)

    def _genCandDirs(self):
        unit = 2.0 / self.lightRes
        uvMap = np.ones((self.lightRes, self.lightRes, 2))
        oneC = np.linspace(-1.0, 1.0, self.lightRes)
        uvMap[:, :, 0] = oneC.reshape((1,-1))/2.0
        uvMap[:, :, 1] = oneC.reshape((-1, 1))*-1.0/2.0

        phis = np.arctan2(uvMap[:,:,1], uvMap[:,:,0]).reshape(-1)
        thetas = np.pi*np.sqrt(uvMap[:,:,1]**2.0 + uvMap[:,:,0]**2.0).reshape(-1)

        self.validMapIds = np.reshape(np.where(thetas < np.pi / 2.0),-1)
        self.chosenIds = np.reshape(np.where(thetas[self.validMapIds] < np.deg2rad(self.deg)),-1)

        dirs = SphToVec(np.column_stack([phis[self.validMapIds], thetas[self.validMapIds]]))
        self.candDirs = dirs

        #fine sample the envmaps to antialias
        self.fineSampleDirs = []
        sampleUnit = unit / (self.nSample + 1)
        for ix in range(self.nSample):
            for iy in range(self.nSample):
                uvMap = np.ones((self.lightRes, self.lightRes, 2))
                oneX = np.linspace(-1.0 - unit/2.0 + (ix+1)*sampleUnit, 1.0 + unit/2.0 - (self.nSample - ix)*sampleUnit, self.lightRes)
                oneY = np.linspace(-1.0 - unit/2.0 + (iy + 1) * sampleUnit, 1.0 + unit/2.0 - (self.nSample - iy) * sampleUnit, self.lightRes)
                uvMap[:, :, 0] = oneX.reshape((1, -1)) / 2.0
                uvMap[:, :, 1] = oneY.reshape((-1, 1)) * -1.0 / 2.0

                curphis = (np.arctan2(uvMap[:, :, 1], uvMap[:, :, 0]).reshape(-1))[self.validMapIds]
                curthetas = (np.pi * np.sqrt(uvMap[:, :, 1] ** 2.0 + uvMap[:, :, 0] ** 2.0).reshape(-1))[self.validMapIds]

                # nonV = np.reshape(np.where(curthetas >= np.pi / 2.0),-1)
                # curphis[nonV] = phis[nonV]
                # curthetas[nonV] = thetas[nonV]

                curDirs = SphToVec(np.column_stack([curphis, curthetas]))
                if type(self.flipOpt) != type(None):
                    if self.flipOpt[0] == 1:
                        curDirs[:, 0] *= -1
                    if self.flipOpt[1] == 1:
                        curDirs[:, 1] *= -1
                self.fineSampleDirs.append(curDirs)



        return dirs

    def _genFrameMaps(self):
        weightMaps = []

        rotAxis = (0,1.0,0)

        candDirs = self.candDirs.copy()
        if type(self.flipOpt) != type(None):
            if self.flipOpt[0] == 1:
                candDirs[:,0] *= -1
            if self.flipOpt[1] == 1:
                candDirs[:,1] *= -1

        for i in range(self.rotRes):
            deg = 360.0 / self.rotRes * i


            rotDirs = rotateVector(candDirs, rotAxis, np.deg2rad(deg))

            r = (1.0/np.pi) * np.arccos(rotDirs[:,2]) / ((1.0-rotDirs[:,2]**2)**0.5)

            pos = ((np.column_stack([rotDirs[:, 0] * r, -rotDirs[:, 1] * r]) * 0.5 + 0.5) * self.envRes)
            color = subPixels(self.envMap, pos[:,0], pos[:,1])

            nS = 1
            for iS in range(len(self.fineSampleDirs)):
                onerotDirs = rotateVector(self.fineSampleDirs[iS], rotAxis, np.deg2rad(deg))

                r = (1.0 / np.pi) * np.arccos(onerotDirs[:, 2]) / ((1.0 - onerotDirs[:, 2] ** 2) ** 0.5)

                pos = ((np.column_stack([onerotDirs[:, 0] * r, -onerotDirs[:, 1] * r]) * 0.5 + 0.5) * self.envRes)


                onecolor = subPixels(self.envMap, pos[:, 0], pos[:, 1])
                color += onecolor
                nS += 1
            color/=nS


            #solidAngleRate dw = sin_theta d_theta d_phi = sin_theta Jacob[d_theta/du, d_phi/dv]dudv
            #solidAnlgeRate = np.sin(curthetas)
            solidAnlgeRate = 0.001
            color *=  self.imgRate * solidAnlgeRate

            weightMaps.append(color)

        return weightMaps, False

    def _genLightImgs(self, outFolder, candDirs, weightMaps):

        weights, bSparse = weightMaps
        lightImg = (np.zeros((self.lightRes, self.lightRes, 3)) * 255).reshape(-1,3)

        if self.chosenIds == None:
            chosenIds = range(len(self.validMapIds))
        else:
            chosenIds = self.chosenIds
        print "choosing ", len(chosenIds), "from ", len(self.validMapIds)

        for iw, weight in enumerate(weights):

            color = weight / 0.01



            gc = np.power(color, 1.0/2.2)*255.0
            lightImg[self.validMapIds[chosenIds]] = gc[chosenIds]
            lightImg[lightImg>255] = 255


            Image.fromarray(lightImg.reshape((self.lightRes, self.lightRes, 3)).astype(np.uint8)).save(outFolder+"/%04d.png"%iw)


# render the scene with a rotating directional light
class CircleAnimator(Animator):
    def __init__(self, relightor, inImgs, inDirs, candFolder=None):

        super(CircleAnimator, self).__init__(relightor, inImgs, inDirs, candFolder)
    def _genCandDirs(self):
        degs = []
        for i in range(self.res):
            theta  = self.centerDeg
            phi = 360.0 / self.res * i
            degs.append([phi, theta])

        return SphToVec(np.deg2rad(degs))

    def _genFrameMaps(self):
        weightMaps = []
        for i in range(self.res):
            weightMaps.append([[i], [1.0,1.0,1.0]])

        return weightMaps, True

    def makeLightImgs(self, outFolder, outPrefix="", deg=30, res=200 ):
        self.centerDeg = deg
        self.res = res

        super(CircleAnimator, self).makeLightImgs(outFolder, outPrefix="circle_" + outPrefix)

    def animate(self, outFolder, restorePath, outPrefix="", deg=30, res=200, **kwargs):
        self.centerDeg = deg
        self.res = res

        super(CircleAnimator, self).animate(outFolder, restorePath, outPrefix="circle_"+outPrefix, **kwargs)

    def _genLightImgs(self, outFolder, candDirs, weightMaps):

        weights, bSparse = weightMaps
        lightImg = np.ones((257, 257, 3)) * 255
        lightImg = cv2.circle(lightImg, (128, 128), 128, (0, 0, 0), -1)

        for corner in self.inDirs:
            lightCoefs = (corner[:2] * (1,-1) * 0.5 + 0.5) * 257
            p0 = (lightCoefs - 5.0).astype(int)
            p1 = p0 + 10

            lightImg = cv2.rectangle(lightImg, (p0[0], p0[1]), (p1[0], p1[1]), (128, 128, 128), -1)


        for iw, weight in enumerate(weights):
            dirs = candDirs[weight[0]]
            colors = (np.reshape(weight[1], (-1,3))*255.0).astype(int)
            curImg = lightImg.copy()
            for id, corner in enumerate(dirs):
                lightCoefs = (corner[:2] * (1,-1) * 0.5 + 0.5) * 257
                p0 = (lightCoefs - 5.0).astype(int)
                p1 = p0 + 10
                curImg = cv2.rectangle(curImg, (p0[0], p0[1]), (p1[0], p1[1]), (0, 255, 255), -1)

            cv2.imwrite(outFolder+"/%04d.png"%iw, curImg)

# render the scene with three lights, which have red, green blue color seperately.
class RGBCircleAnimator(CircleAnimator):
    def __init__(self, relightor, inImgs, inDirs):
        super(CircleAnimator, self).__init__(relightor, inImgs, inDirs)

    def _genFrameMaps(self):
        weightMaps = []
        startIds = np.asarray([0, self.res/3, self.res/3 * 2], np.int)
        for i in range(self.res):
            weightMaps.append([(startIds + i)%self.res, [[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]])
        return weightMaps, True

    def makeLightImgs(self, outFolder, outPrefix="", deg=30, res=200 ):

        super(RGBCircleAnimator, self).makeLightImgs(outFolder, outPrefix="RGB_" + outPrefix, deg=deg, res=res)

    def animate(self, outFolder, restorePath, outPrefix="", deg=30, res=200 ):


        super(RGBCircleAnimator, self).animate(outFolder, restorePath, outPrefix="RGB_"+outPrefix, deg=deg, res=res)
