import numpy as np
import cv2
import os
from ..Common import *

class Shape(object):
    def __init__(self):
        self.points = []
        self.uvs = []
        self.faces = []
        self.facesUV = []
        self.matNames = []
        self.matStartId = []

    def genShape(self):
        self.points = np.reshape([], (-1,3)).astype(float)
        self.uvs = np.reshape([], (-1,2)).astype(float)
        self.faces = np.reshape([], (-1,3)).astype(int)
        self.facesUV = np.reshape([], (-1,3)).astype(int)
        self.matNames = []
        self.matStartId = []

    def genObj(self, filePath, bMat = False):
        if len(self.faces) == 0:
            print "no mesh"
            return False
        with open(filePath, "w") as f:
            #write v
            for point in self.points:
                f.write("v %f %f %f\n"%(point[0], point[1], point[2]))
            # write uv
            for uv in self.uvs:
                f.write("vt %f %f\n" % (uv[0], uv[1]))
            #write face
            # f.write("usemtl mat_%d\n"%matId)
            if not bMat:
                for i,face in enumerate(self.faces):
                    f.write("f %d/%d %d/%d %d/%d\n" %
                            (face[0], self.facesUV[i][0], face[1], self.facesUV[i][1], face[2], self.facesUV[i][2]))
            else:
                for im in range(len(self.matStartId)):
                    f.write("usemtl %s\n"%self.matNames[im])
                    if im == len(self.matStartId) - 1:
                        endId = len(self.faces)
                    else:
                        endId = self.matStartId[im+1]
                    for i in range(self.matStartId[im], endId):
                        f.write("f %d/%d %d/%d %d/%d\n" %
                                (self.faces[i][0], self.facesUV[i][0], self.faces[i][1], self.facesUV[i][1], self.faces[i][2], self.facesUV[i][2]))

        return True

    def genMatList(self, filePath):
        with open(filePath, "w") as f:
            for matname in self.matNames:
                f.write("%s\n"%matname)
    def genInfo(self, filePath):
        with open(filePath, "w") as f:
            minP = np.min(self.points, axis=0)
            maxP = np.max(self.points, axis=0)
            print minP, maxP
            f.write("%f %f %f\n" % (minP[0], minP[1], minP[2]))
            f.write("%f %f %f\n" % (maxP[0], maxP[1], maxP[2]))


    def translate(self, translation):
        self.points += translation

    def rotate(self, axis, degAngle):
        self.points = rotateVector(self.points, axis, np.deg2rad(degAngle))

    def reCenter(self):
        minP = np.min(self.points, 0)
        maxP = np.max(self.points, 0)

        center = 0.5*minP + 0.5*maxP

        self.translate(-center)

    def addShape(self, otherShape):
        curPN = len(self.points)
        curUN = len(self.uvs)
        curFN = len(self.faces)
        if curPN == 0:
            self.points = np.copy(otherShape.points)
            self.uvs = np.copy(otherShape.uvs)
            self.faces = np.copy(otherShape.faces)
            self.facesUV = np.copy(otherShape.facesUV)
            self.matNames += otherShape.matNames
            self.matStartId = np.append(self.matStartId, otherShape.matStartId + curFN).astype(int)
        else:
            self.points = np.row_stack([self.points, otherShape.points])
            self.uvs = np.row_stack([self.uvs, otherShape.uvs])
            self.faces = np.row_stack([self.faces, otherShape.faces+curPN])
            self.facesUV = np.row_stack([self.facesUV, otherShape.facesUV+curUN])
            self.matNames += otherShape.matNames
            self.matStartId = np.append(self.matStartId, otherShape.matStartId+curFN).astype(int)

    def _addMorphCircle(self, center=(0,0,0), axisA = 1.0, axisB = 1.0, X=(1,0,0), Z=(0,0,1), circelRes = (50, 100), matName = "mat"):
        X = np.reshape(X,3)
        Z = np.reshape(Z,3)
        Y = np.cross(Z, X)
        startPId = len(self.points)
        startUId = len(self.uvs)
        startFaceId = len(self.faces)

        center = np.reshape(center, 3)
        points = []
        uvs = []
        points.append(center)
        uvs.append((0.5, 0.5))

        # create points
        for iy in range(1, circelRes[0]):
            for ix in range(circelRes[1]):
                ra = float(axisA) * iy / (circelRes[0] - 1)
                rb = float(axisB) * iy / (circelRes[0] - 1)


                phi = float(ix) /(circelRes[1]) * 2.0 * np.pi

                x = ra *  np.cos(phi)
                y = rb *  np.sin(phi)


                p = x*X + y*Y + center
                points.append(p)
        # create uvs
        for iy in range(1, circelRes[0]):
            for ix in range(circelRes[1]):
                ra = float(axisA) * iy / (circelRes[0] - 1)
                rb = float(axisB) * iy / (circelRes[0] - 1)

                phi = float(ix) / (circelRes[1]) * 2.0 * np.pi

                x = ra * np.cos(phi)
                y = rb * np.sin(phi)

                r = (x**2 + y**2)**0.5
                rl = ((axisA*np.cos(phi))**2 + (axisB*np.sin(phi))**2)**0.5

                if phi > np.pi*7.0/4.0 or phi <= np.pi / 4.0:
                    ul = 1.0
                    vl = -0.5 * np.tan(phi) + 0.5
                elif phi > np.pi / 4.0 and phi <= np.pi * 3.0 / 4.0:
                    vl = 0.0
                    ul = 1.0 / np.tan(phi) * 0.5 + 0.5
                elif phi > np.pi * 3.0 / 4.0 and phi <= np.pi * 5.0 / 4.0:
                    ul = 0.0
                    vl = 0.5 * np.tan(phi) + 0.5
                else: #phi > np.pi * 5.0 / 4.0 and phi <= np.pi * 7.0 / 4.0:
                    vl = 1.0
                    ul = -1.0 / np.tan(phi) * 0.5 + 0.5
                u = (ul-0.5) / rl * r + 0.5
                v = (vl-0.5) / rl * r + 0.5
                uvs.append((u, v))

        if startPId == 0:
            self.points = np.reshape(points, (-1, 3))
            self.uvs = np.rehsape(uvs, (-1, 2))
        else:
            self.points = np.row_stack([self.points, points])
            self.uvs = np.row_stack([self.uvs, uvs])

        # create faces
        tempFaces = []
        for iy in range(circelRes[0] - 1):
            for ix in range(circelRes[1]):
                if iy == 0:
                    curId = 1
                    rightId = 1
                    bottomId = 1 + ix + 1
                    if ix == circelRes[1] - 1:
                        rightBottomId = 1 + 1
                    else:
                        rightBottomId = 1 + ix + 1 + 1
                else:
                    curId = 1 + (iy - 1) * circelRes[1] + ix + 1
                    bottomId = 1 + (iy) * circelRes[1] + ix + 1
                    if ix == circelRes[1] - 1:
                        rightId = 1 + (iy - 1) * circelRes[1] + 1
                        rightBottomId = 1 + (iy) * circelRes[1] + 1
                    else:
                        rightId = 1 + (iy - 1) * circelRes[1] + ix + 1 + 1
                        rightBottomId = 1 + (iy) * circelRes[1] + ix + 1 + 1
                if iy != 0:
                    tempFaces.append((curId, rightBottomId, rightId))
                tempFaces.append((curId, bottomId, rightBottomId))

        tempFaces = np.reshape(tempFaces, (-1,3))

        if len(self.faces) == 0:
            self.faces = tempFaces.copy()
            self.facesUV = tempFaces.copy()
            self.matStartId = self.matStartId = np.asarray([0],int)
        else:
            self.faces = np.row_stack([self.faces, tempFaces+startPId])
            self.facesUV = np.row_stack([self.facesUV, tempFaces+startUId])
            self.matStartId = np.append(self.matStartId, [startFaceId])

        self.matNames.append(matName)




class HeightFieldCreator:
    def __init__(self, initSize = (5,5), maxHeight = (-0.2,0.2), bFixCorner = True):
        self.initSize = initSize
        self.bFixCorner = bFixCorner
        self.initNum = self.initSize[0]*self.initSize[1]
        self.maxHeight = maxHeight
        self.heightField = None

    def __initializeHeigthField(self):
        heights = np.random.uniform(self.maxHeight[0], self.maxHeight[1], self.initNum)
        # if self.bFixCorner:
        #     heights[0] = heights[self.initSize[1]-1] = heights[(self.initSize[0]-1)*self.initSize[1]] = heights[-1] = 0
        initHeightField = heights.reshape(self.initSize)
        self.initHeightField = initHeightField
        return initHeightField

    def genHeightField(self, targetSize = (36, 36)):
        halfSize = (targetSize[0]/6*5, targetSize[1]/6*5)
        if halfSize[0] < self.initSize[0] or halfSize[1] < self.initSize[1]:
            print "target size should be double as init size"
            return None
        initHeight = self.__initializeHeigthField()
        if self.bFixCorner:
            bounder = np.zeros((self.initSize[0]+2, self.initSize[1]+2))
            bounder[1:-1, 1:-1] = initHeight
            initHeight = bounder
        heightField_half = cv2.resize(initHeight, halfSize, interpolation=cv2.INTER_CUBIC)#
        if self.bFixCorner:
            bounder = np.zeros(halfSize)
            bounder[1:-1, 1:-1] = heightField_half[1:-1, 1:-1]
            initHeight = bounder
        heightField = cv2.resize(initHeight, targetSize)  #
        self.heightField = heightField
        self.targetSize = targetSize
        return heightField

    def genObj(self, filePath):
        if self.heightField == None:
            print "no generated height fields"
            return False


        with open(filePath, "w") as f:
            #write v
            for iy in range(self.targetSize[0]):
                for ix in range(self.targetSize[1]):
                    f.write("v %f %f %f\n"%
                            (float(ix)/(self.targetSize[1]-1),
                             float(iy)/(self.targetSize[0]-1),
                             self.heightField[iy, ix]))
            #write f
            for iy in range(self.targetSize[0]-1):
                for ix in range(self.targetSize[1]-1):
                    curId = iy * self.targetSize[1] + ix + 1
                    rightId = iy * self.targetSize[1] + ix + 1 +1
                    bottomId = (iy+1) * self.targetSize[1] + ix+1
                    rightBottomId = (iy+1) * self.targetSize[1] + ix + 1+1
                    f.write("f %d %d %d\n"%
                            (curId, rightBottomId, rightId))
                    f.write("f %d %d %d\n" %
                            (curId, bottomId, rightBottomId))
        return True

class Ellipsoid(Shape):
    # meshRes: rows x columns, latitude res x longitude res
    def __init__(self, a = 1.0, b = 1.0, c = 1.0, meshRes = (50, 100)):
        super(Ellipsoid, self).__init__()
        if meshRes[1] % 2 != 0:
            print "WARN: longitude res is supposed to be even"
        self.axisA = a
        self.axisB = b
        self.axisC = c
        self.meshRes = meshRes

        self.numPoints = (self.meshRes[0] - 2) * self.meshRes[1] + 2


    def genShape(self, matName = "mat"):
        super(Ellipsoid, self).__init__()


        self.points.append((0,0,self.axisC))
        self.uvs.append((0,0))


        #create points
        for iy in range(1,self.meshRes[0]-1):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0]-1)
                u = float(ix) / (self.meshRes[1]/2)

                theta = np.pi/2.0 - v * np.pi
                phi = u * np.pi

                x = self.axisA * np.cos(theta) * np.cos(phi)
                y = self.axisB * np.cos(theta) * np.sin(phi)
                z = self.axisC * np.sin(theta)

                self.points.append((x,y,z))
        self.points.append((0, 0, -self.axisC))

        #create uvs
        for iy in range(1, self.meshRes[0] - 1):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)
                if u > 1.0:
                    u = 2.0 - u
                self.uvs.append((u,v))
        self.uvs.append((1.0,1.0))

        #create faces
        for iy in range(self.meshRes[0]-1):
            for ix in range(self.meshRes[1]):
                if iy == 0:
                    curId = 1
                    rightId = 1
                    bottomId = 1 + ix + 1
                    if ix == self.meshRes[1] - 1:
                        rightBottomId = 1 + 1
                    else:
                        rightBottomId = 1+ ix + 1 + 1
                elif iy == self.meshRes[0]-2:
                    curId = 1 + (iy-1) * self.meshRes[1] + ix + 1
                    bottomId = 1 + (iy) * self.meshRes[1] + 1
                    if ix == self.meshRes[1] - 1:
                        rightId = 1 + (iy-1) * self.meshRes[1] + 1
                    else:
                        rightId = 1 + (iy-1) * self.meshRes[1] + ix + 1 + 1
                    rightBottomId = 1 + (iy) * self.meshRes[1] + 1
                else:
                    curId = 1 + (iy - 1) * self.meshRes[1] + ix + 1
                    bottomId = 1 + (iy) * self.meshRes[1] + ix + 1
                    if ix == self.meshRes[1] - 1:
                        rightId = 1 + (iy - 1) * self.meshRes[1] + 1
                        rightBottomId = 1 + (iy) * self.meshRes[1] + 1
                    else:
                        rightId = 1 + (iy - 1) * self.meshRes[1] + ix + 1 + 1
                        rightBottomId = 1 + (iy) * self.meshRes[1]+ ix + 1 + 1
                if iy != 0:
                    self.faces.append((curId, rightBottomId, rightId))
                if iy != self.meshRes[0]-2:
                    self.faces.append((curId, bottomId, rightBottomId))

        self.points = np.reshape(self.points, (-1,3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1,2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.copy(self.faces)
        self.matNames = [matName]
        self.matStartId = np.asarray([0],int)

    def applyHeightField(self, heightFields):
        if len(self.points) == 0:
            print "no points"
            return False
        if len(heightFields.shape) != 2 and len(heightFields.shape) != 3:
            print "wrong shape of heightfiels"
            return False
        if len(heightFields.shape) == 3:
            heightField = heightFields[0]


        for i,point in enumerate(self.points):
            uv = self.uvs[i]
            normal = np.reshape(point,-1) / (self.axisA, self.axisB, self.axisC)
            normal = normal / np.linalg.norm(normal)
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])#cv2.getRectSubPix(heightField, (1,1), (xy[0], xy[1]))

            self.points[i] = point + normal * h


class Cube(Shape):
    """faces:
    front: c = 1
    back: c = -1
    left: a = -1
    right: a = 1
    up: b = 1
    down: b = -1

    """
    def __init__(self, a=1.0, b=1.0, c=1.0, faceRes=(50, 50)):
        super(Cube,self).__init__()
        self.axisA = a
        self.axisB = b
        self.axisC = c
        self.faceRes = faceRes
        self.pointNumPerFace = faceRes[0] * faceRes[1]

        self.numPoints = (self.faceRes[0]) * self.faceRes[1] * 6


    def genShape(self, matName = "mat"):
        super(Cube, self).__init__()


        #uvs
        for iy in range(self.faceRes[0]):
            for ix in range(self.faceRes[1]):
                u = float(ix) / (self.faceRes[1] - 1)
                v = float(iy) / (self.faceRes[0] - 1)
                self.uvs.append((u,v))
        self.uvs = np.reshape(self.uvs, (-1,2))
        #face:
        #oneFace:
        oneFaces = []
        for iy in range(self.faceRes[0] - 1):
            for ix in range(self.faceRes[1] - 1):
                curId = iy * self.faceRes[1] + ix + 1
                rightId = iy * self.faceRes[1] + ix + 1 + 1
                bottomId = (iy + 1) * self.faceRes[1] + ix + 1
                rightBottomId = (iy + 1) * self.faceRes[1] + ix + 1 + 1

                oneFaces.append((curId, rightBottomId, rightId))
                oneFaces.append((curId, bottomId, rightBottomId))
        oneFaces = np.reshape(oneFaces, (-1,3)).astype(int)
        self.faces = np.row_stack([oneFaces,
                                      oneFaces + self.pointNumPerFace,
                                      oneFaces + self.pointNumPerFace*2,
                                      oneFaces + self.pointNumPerFace*3,
                                      oneFaces + self.pointNumPerFace*4,
                                      oneFaces + self.pointNumPerFace*5])
        self.facesUV = np.row_stack([oneFaces, oneFaces, oneFaces, oneFaces, oneFaces, oneFaces])

        #points
        #front
        for uv in self.uvs:
            xy = uv * (self.axisA, -self.axisB) * 2.0 + (-self.axisA, self.axisB)
            point = (xy[0], xy[1], self.axisC)
            self.points.append(point)

        # back
        for uv in self.uvs:
            xy = uv * (self.axisA, self.axisB) * 2.0 + (-self.axisA, -self.axisB)
            point = (xy[0], xy[1], -self.axisC)
            self.points.append(point)

        #left
        for uv in self.uvs:
            zy = uv * (self.axisC, -self.axisB) * 2.0 + (-self.axisC, self.axisB)
            point = (-self.axisA, zy[1], zy[0])
            self.points.append(point)

        # right
        for uv in self.uvs:
            zy = uv * (-self.axisC, -self.axisB) * 2.0 + (self.axisC, self.axisB)
            point = (self.axisA, zy[1], zy[0])
            self.points.append(point)

        # up
        for uv in self.uvs:
            xz = uv * (self.axisA, self.axisC) * 2.0 + (-self.axisA, -self.axisC)
            point = (xz[0], self.axisB, xz[1])
            self.points.append(point)

        # down
        for uv in self.uvs:
            xz = uv * (self.axisA, -self.axisC) * 2.0 + (-self.axisA, self.axisC)
            point = (xz[0], -self.axisB, xz[1])
            self.points.append(point)

        self.points = np.reshape(self.points, (-1, 3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1, 2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.reshape(self.facesUV, (-1, 3)).astype(int)
        self.matNames = ["%s_%d"%(matName, 0),
                         "%s_%d"%(matName, 1),
                         "%s_%d"%(matName, 2),
                         "%s_%d"%(matName, 3),
                         "%s_%d"%(matName, 4),
                         "%s_%d"%(matName, 5)]
        self.matStartId = np.asarray([0,
                           self.pointNumPerFace,
                           self.pointNumPerFace*2,
                           self.pointNumPerFace*3,
                           self.pointNumPerFace*4,
                           self.pointNumPerFace*5],int)

    # def genObj(self, filePath):
    #     if len(self.faces) == 0:
    #         print "no mesh"
    #         return False
    #     with open(filePath, "w") as f:
    #         #write v
    #         for point in self.points:
    #             f.write("v %f %f %f\n"%(point[0], point[1], point[2]))
    #         # write uv
    #         for uv in self.uvs:
    #             f.write("vt %f %f\n" % (uv[0], uv[1]))
    #         #write face
    #         for i,face in enumerate(self.faces):
    #             f.write("f %d/%d %d/%d %d/%d\n" %
    #                     (face[0], (face[0]-1)%self.pointNumPerFace+1,
    #                      face[1], (face[1]-1)%self.pointNumPerFace+1,
    #                      face[2], (face[2]-1)%self.pointNumPerFace+1))
    #     return True


    def applyHeightField(self, heightFields):
        if len(self.points) == 0:
            print "no points"
            return False
        if len(heightFields.shape) != 2 and len(heightFields.shape) != 3:

            print "wrong shape of heightfiels"
            return False

        if len(heightFields.shape) == 2:
            newH = []
            for i in range(6):
                newH.append(heightFields)
            heightFields = newH
        else:
            if heightFields.shape[0] < 6:
                newH = []
                for i in range(6):
                    if i < heightFields.shape[0]:
                        newH.append(heightFields[i])
                    else:
                        newH.append(heightFields[-1])
                heightFields = newH

        # modify points
        # front
        heightField = heightFields[0]
        normal = np.asarray((0,0,1))
        offSet = 0
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i+offSet] + h * normal


        # back
        heightField = heightFields[1]
        normal = np.asarray((0, 0, -1))
        offSet = self.pointNumPerFace*1
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

        # left
        heightField = heightFields[2]
        normal = np.asarray((-1, 0, 0))
        offSet = self.pointNumPerFace * 2
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

        # right
        heightField = heightFields[3]
        normal = np.asarray((1, 0, 0))
        offSet = self.pointNumPerFace * 3
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

        # up
        heightField = heightFields[4]
        normal = np.asarray((0, 1, 0))
        offSet = self.pointNumPerFace * 4
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

        # down
        heightField = heightFields[5]
        normal = np.asarray((0, -1, 0))
        offSet = self.pointNumPerFace * 5
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

class Cylinder(Shape):
    def __init__(self, a=1.0, b=1.0, c=1.0, meshRes=(50, 150), radiusRes = 20):
        super(Cylinder,self).__init__()

        self.axisA = a
        self.axisB = b
        self.axisC = c
        self.meshRes = meshRes



    def genShape(self, matName = "mat"):
        super(Cylinder, self).__init__()


        # create points
        for iy in range(self.meshRes[0]):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)

                phi = u * np.pi

                x = self.axisA * np.cos(phi)
                y = self.axisB * np.sin(phi)
                z = self.axisC - self.axisC * v * 2.0

                self.points.append((x, y, z))



        # create uvs
        for iy in range(self.meshRes[0]):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)
                if u > 1.0:
                    u = 2.0 - u
                self.uvs.append((u, v))


        # create faces
        for iy in range(self.meshRes[0]-1):
            for ix in range(self.meshRes[1]):

                curId = iy * self.meshRes[1] + ix + 1
                bottomId = (iy + 1) * self.meshRes[1] + ix + 1
                if ix == self.meshRes[1] - 1:
                    rightId = iy * self.meshRes[1] + 1
                    rightBottomId = (iy + 1) * self.meshRes[1] + 1
                else:
                    rightId = (iy) * self.meshRes[1] + ix + 1 + 1
                    rightBottomId = (iy + 1) * self.meshRes[1] + ix + 1 + 1

                self.faces.append((curId, rightBottomId, rightId))
                self.faces.append((curId, bottomId, rightBottomId))

        # self.points.append((0, 0, self.axisC))
        # self.points.append((0, 0, -self.axisC))
        # self.uvs.append((0.5, 0))
        # self.uvs.append((0.5, 1))
        # #up
        # centerid = self.meshRes[0]*self.meshRes[1] + 1
        # for ix in range(self.meshRes[1]-1):
        #     self.faces.append((centerid, ix+1, ix+2))
        # self.faces.append((centerid, self.meshRes[1], 1))
        #
        # # #down
        # centerid = self.meshRes[0] * self.meshRes[1] + 2
        # for ix in range(self.meshRes[1] - 1):
        #     self.faces.append((centerid, ix+(self.meshRes[0] - 1)*self.meshRes[1] + 2, ix+(self.meshRes[0] - 1)*self.meshRes[1]+1))
        # self.faces.append((centerid, (self.meshRes[0] - 1)*self.meshRes[1]+1, (self.meshRes[0])*self.meshRes[1]))

        self.points = np.reshape(self.points, (-1, 3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1, 2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.copy(self.faces)
        self.matNames = ["%s_0"%matName]
        self.matStartId = self.matStartId = np.asarray([0],int)

        self._addMorphCircle((0, 0, self.axisC), self.axisA, self.axisB, X=(1,0,0), Z=(0,0,1),
                             circelRes=[self.meshRes[0]/2, self.meshRes[1]], matName="%s_1"%matName)

        self._addMorphCircle((0, 0, -self.axisC), self.axisA, self.axisB, X=(1, 0, 0), Z=(0, 0, -1),
                             circelRes=[self.meshRes[0]/2, self.meshRes[1]], matName="%s_2" % matName)



    def applyHeightField(self, heightFields):
        if len(self.points) == 0:
            print "no points"
            return False
        if len(heightFields.shape) != 2 and len(heightFields.shape) != 3:

            print "wrong shape of heightfiels"
            return False

        if len(heightFields.shape) == 2:
            newH = []
            for i in range(3):
                newH.append(heightFields)
            heightFields = newH
        else:
            if heightFields.shape[0] < 3:
                newH = []
                for i in range(3):
                    if i < heightFields.shape[0]:
                        newH.append(heightFields[i])
                    else:
                        newH.append(heightFields[-1])
                heightFields = newH

        heightField = heightFields[0]
        i = 0
        for iy in range(self.meshRes[0]):
            for ix in range(self.meshRes[1]):
                u = float(ix) / (self.meshRes[1] / 2)

                phi = u * np.pi

                x = self.axisA * np.cos(phi)
                y = self.axisB * np.sin(phi)

                normal = np.reshape((x,y,0),-1)/ (self.axisA, self.axisB, self.axisC)
                normal = normal / np.linalg.norm(normal)
                xy = self.uvs[i] * (heightField.shape[1], heightField.shape[0])
                h = subPix(heightField, xy[0], xy[1])  # cv2.getRectSubPix(heightField, (1,1), (xy[0], xy[1]))

                self.points[i] += normal * h
                i+=1

        heightField = heightFields[1]
        circelRes = [self.meshRes[0] / 2, self.meshRes[1]]
        for iy in range(circelRes[0] - 1):
            for ix in range(circelRes[1]):
                normal = np.reshape((0, 0, 1),-1)
                xy = self.uvs[i] * (heightField.shape[1], heightField.shape[0])
                h = subPix(heightField, xy[0], xy[1])  # cv2.getRectSubPix(heightField, (1,1), (xy[0], xy[1]))
                self.points[i] += normal * h
                i+=1

        heightField = heightFields[2]
        circelRes = [self.meshRes[0] / 2, self.meshRes[1]]
        for iy in range(circelRes[0] - 1):
            for ix in range(circelRes[1]):
                normal = np.reshape((0, 0, -1), -1)
                xy = self.uvs[i] * (heightField.shape[1], heightField.shape[0])
                h = subPix(heightField, xy[0], xy[1])  # cv2.getRectSubPix(heightField, (1,1), (xy[0], xy[1]))
                self.points[i] += normal * h
                i += 1



class MultiShape(Shape):
    """
    0: ellipsoid
    1: cube
    2: cylinder

    """
    def __init__(self,
                 numShape = 6, smoothPossibility = 0.1, axisRange = (0.25, 2.0), heightRangeRate = (0, 0.2),
                 translateRangeRate = (0, 0.5), rotateRange = (0, 180)):
        super(MultiShape, self).__init__()
        self.numShape = numShape
        self.smoothPossibility = smoothPossibility
        self.axisRange = axisRange
        self.heightRangeRate = heightRangeRate
        self.translateRangeRate = translateRangeRate
        self.rotateRange = rotateRange
    def genShape(self):
        super(MultiShape, self).__init__()


        for iS in range(self.numShape):
            rp = np.random.permutation(3)
            axisVals = np.random.uniform(self.axisRange[0], self.axisRange[1], 3)
            hfs = []
            minA = axisVals.min()*2.0
            maxA = axisVals.max()*2.0
            print minA, maxA
            maxH = np.random.uniform(self.heightRangeRate[0]*minA, self.heightRangeRate[1]*minA, 6)
            translation = np.random.uniform(self.translateRangeRate[0]*maxA, self.translateRangeRate[1]*maxA, 3)
            translation1 = np.random.uniform(self.translateRangeRate[0] * maxA, self.translateRangeRate[1] * maxA, 3)
            rotation  = np.random.uniform(self.rotateRange[0], self.rotateRange[1], 3)
            rotation1 = np.random.uniform(self.rotateRange[0], self.rotateRange[1], 3)
            for ih in range(6):
                smoothR = np.random.uniform(0,1,1)[0]
                if smoothR <= self.smoothPossibility or maxH[ih] == 0:
                    hf = np.zeros((36,36))
                else:
                    hfg = HeightFieldCreator(maxHeight=(-maxH[ih], maxH[ih]))
                    hf = hfg.genHeightField()
                hfs.append(hf)
            hfs = np.reshape(hfs, (6,) + hf.shape)

            if rp[0] == 0:
                subShape = Ellipsoid(axisVals[0], axisVals[1], axisVals[2])
            elif rp[0] == 1:
                subShape = Cube(axisVals[0], axisVals[1], axisVals[2])
            elif rp[0] == 2:
                subShape = Cylinder(axisVals[0], axisVals[1], axisVals[2])

            subShape.genShape(matName="mat_shape%d"%iS)
            subShape.applyHeightField(hfs)


            subShape.rotate((1, 0, 0), rotation[0])
            subShape.rotate((0, 1, 0), rotation[1])
            subShape.rotate((0, 0, 1), rotation[2])
            subShape.translate(translation)

            if iS != 0:
                self.rotate((1, 0, 0), rotation1[0])
                self.rotate((0, 1, 0), rotation1[1])
                self.rotate((0, 0, 1), rotation1[2])
                self.translate(translation1)

            self.addShape(subShape)

        self.reCenter()


def createShapes(outFolder, shapeNum, subObjNum = 6):
    if not os.path.isdir(outFolder):
        os.makedirs(outFolder)

    for i in range(shapeNum):
        ms = MultiShape(subObjNum)
        subFolder = outFolder + "/Shape__%d"%i
        if not os.path.isdir(subFolder):
            os.makedirs(subFolder)
        ms.genShape()
        ms.genObj(subFolder + "/object.obj", bMat=True)
        ms.genMatList(subFolder + "/object.txt")
        ms.genInfo(subFolder + "/object.info")

def createVarObjShapes(outFolder, shapeNum, subObjNums = [1,2,3,4,5,6,7,8,9], subObjPoss = [1,2,3,7,10,7,3,2,1]):
    if len(subObjNums) != len(subObjPoss):
        print "In correct obj num distribution"
        exit()

    if not os.path.isdir(outFolder):
        os.makedirs(outFolder)


    subOjbBound = np.reshape(subObjPoss, -1).astype(float)
    subOjbBound = subOjbBound / np.sum(subOjbBound)
    subOjbBound = np.cumsum(subOjbBound)

    if subOjbBound[-1] != 1.0:
        print "in correct bound"
        subOjbBound[-1] = 1.0


    counts = np.zeros(len(subObjNums))

    chooses = np.random.uniform(0, 1.0, shapeNum)
    for i in range(shapeNum):
        choose = chooses[i]
        subObjNum = subObjNums[-1]
        for iO in range(len(subOjbBound)):
            if choose < subOjbBound[iO]:
                subObjNum = subObjNums[iO]
                counts[iO] += 1
                break

        print i, subObjNum
        ms = MultiShape(subObjNum)
        subFolder = outFolder + "/Shape__%d"%i
        if not os.path.isdir(subFolder):
            os.makedirs(subFolder)
        ms.genShape()
        ms.genObj(subFolder + "/object.obj", bMat=True)
        ms.genMatList(subFolder + "/object.txt")
        ms.genInfo(subFolder + "/object.info")
    print counts

