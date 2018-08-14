from relight.animation.Animator import *
from relight.CNNRelighting.RelightingNetwork import *
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




if __name__ == "__main__":


    candDegs = [30, 45, 60, 90]

    parser = argparse.ArgumentParser(description= "relight test data")
    parser.add_argument('deg', choices=candDegs, type=int, default=90, help="Specify the degree of the relighting cone. Corresponding weights and directions will be selected from weightFolder and dirFolder.")
    parser.add_argument('inputNum', choices=[3,4,5,6,7,8], default=5, type=int, help="Specify the number of input samples. Corresponding weights and directions will be selected from weightFolder and dirFolder.")
    parser.add_argument('-i', '--inputFolder', default = "../data/test/mapper_100_sameDir", help="The folder containing the test data.")
    parser.add_argument('-w', '--weightFolder', default = "../trained/refine", help = "The folder containing the trained weights.")
    parser.add_argument('-d', '--dirFolder', default = "../trained/dirs", help = "The folder containing the learned light directions.")
    parser.add_argument('-o', '--outFolder', default= "../out/envRender", help = "Output folder.")
    parser.add_argument('-f', '--envFolder', default= "../envs", help = "The folder containing the environment maps.")
    parser.add_argument('-e', '--envs', default= ["grace_probe.pfm"], nargs='+', help = "The environment maps that will be used to render")
    parser.add_argument('-p', '--saveInputImage', action='store_true', help="A flag to save the input images in an seperate folder.")
    parser.add_argument('-l', '--saveLightImage', action='store_true', help="A flag to save the rotated environment map images.")
    parser.add_argument('-s', '--imageScale', type=float, default=0.5, help="A scalar that will scale the image intensity")
    parser.add_argument('-t', '--targets', metavar='id', type=int, nargs='+', default=[0], help="If specified, only a subset of scenes will be tested.")


    args = parser.parse_args()

    testFolder = args.inputFolder
    weightFolder = args.weightFolder
    outFolder = args.outFolder
    dirFolder = args.dirFolder
    targets = args.targets
    deg = args.deg
    inputNum = args.inputNum
    imageScale = args.imageScale
    bSaveInput = args.saveInputImage
    bMakeLight = args.saveLightImage
    lightProbes = args.envs
    envFolder = args.envFolder

    if not os.path.isfile(weightFolder + "/deg%d_%d.index"%(deg, inputNum)):
        print "no weights for deg%d %d"%(deg, inputNum)
        exit()

    print "testing deg %d input %d"%(deg, inputNum)
    print "using weight from: %s"%(weightFolder)



    outFolder += "/deg%d_%d"%(deg, inputNum)


    inDirs = np.loadtxt(dirFolder + "/deg%d_%d.txt"%(deg, inputNum))


    msCNN = RelightingNet(imgSize=(512, 512), inputNum=inputNum)

    bRecalc = False

    restorePath = weightFolder + "/deg%d_%d"%(deg, inputNum)

    print targets, lightProbes, restorePath, os.path.isfile(restorePath)

    for iP, probe in enumerate(lightProbes):
        env = load_pfm("%s/%s" % (envFolder,probe))
        env = cv2.resize(env, (env.shape[0] / 2, env.shape[0] / 2))
        env = cv2.resize(env, (env.shape[0] / 2, env.shape[0] / 2))
        env = cv2.resize(env, (env.shape[0] / 2, env.shape[0] / 2))

        for target in targets:

            if not os.path.isdir(outFolder +"/%d/frames_env_%d_%s"%(target,deg, probe)):
                os.makedirs(outFolder +"/%d/frames_env_%d_%s"%(target,deg, probe))

            print probe, outFolder
            allDirs = np.load(testFolder+"/Dirs.npy")
            allCoefs = np.load(testFolder+"/Coefs.npy")
            inIds = []
            for ii, oneDir in enumerate(inDirs):
                errs = np.linalg.norm(allDirs - oneDir, axis=1)
                inIds.append(np.argmin(errs))
                print errs[inIds[ii]]
            print inIds
            # exit()
            imgs = []

            #make input images
            for ii, id in enumerate(inIds):
                img = np.asarray(Image.open(testFolder + "/Shape__%d/0/inters/%d_%.3f_%.3f.png" % (target,
                        id,
                        allCoefs[id][0], allCoefs[id][1])), np.uint8)
                imgs.append(img)


                if iP == 0 and bSaveInput:
                    inFolder = outFolder + "/%d/inImgs"%target
                    if not os.path.isdir(inFolder):
                        os.makedirs(inFolder)
                    Image.fromarray(img).save(inFolder+"/%d_%d_%.3f_%.3f.png" % (
                        ii, id,
                        allCoefs[id][0], allCoefs[id][1]))

            if bRecalc:
                candFolder = None
            else:
                candFolder = "/cands"



            ant = EnvFineCircleAnimator(msCNN, imgs, allDirs[inIds], imageScale, nSample=10, candFolder=candFolder)
            ant.animate(outFolder+"/%d"%target,
                        restorePath,
                        envMap=env, lightRes=64, deg=deg, outPrefix="%d_%s"%(deg,probe), bMakeLight=bMakeLight)

