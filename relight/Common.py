
import re
import numpy as np
import  sys

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def rotateVector(vector, axis, angle):
    cos_ang = np.reshape(np.cos(angle),(-1));
    sin_ang = np.reshape(np.sin(angle),(-1));
    vector = np.reshape(vector,(-1,3))
    axis = np.reshape(np.array(axis),(-1,3))
    return vector * cos_ang[:,np.newaxis] + axis*np.dot(vector,np.transpose(axis))*(1-cos_ang)[:,np.newaxis] + np.cross(axis,vector) * sin_ang[:,np.newaxis]

def normalize(x):
    if(len(np.shape(x)) == 1):
        return x/np.linalg.norm(x)
    else:
        return x/np.linalg.norm(x,axis=1)[:,np.newaxis]

def copyFile(src, target):
    with open(src, "rb") as f:
        data = f.read()
    with open(target, "wb") as f:
        f.write(data)

#return Phi(0, 2pi), Theta (0, pi) in rads
def VecToSph(coords):

    coords = np.reshape(coords,(-1,3))

    rads = np.zeros((coords.shape[0],2))

    rads[:,0] = np.arctan2(coords[:,1], coords[:,0])
    rads[rads<0] += 2.0 * np.pi
    rads[:,1] = np.arccos(coords[:,2])

    return rads

#Phi Theta
def SphToVec(coords):
    coords = np.reshape(coords,(-1,2))

    vec = np.zeros((coords.shape[0],3))
    vec[:,0] = np.cos(coords[:,0])*np.sin(coords[:,1])
    vec[:,1] = np.sin(coords[:,0])*np.sin(coords[:,1])
    vec[:,2] = np.cos(coords[:,1])
    return vec

def subPixels(img, xs, ys):
    height = img.shape[0]
    width = img.shape[1]
    xs = np.reshape(xs, -1)
    ys = np.reshape(ys, -1)
    ix0 = xs.astype(int)
    iy0 = ys.astype(int)
    ix1 = ix0+1
    iy1 = iy0+1

    ids = np.reshape(np.where(ix0 < 0), -1)
    if len(ids) > 0:
        ix0[ids]=0
        ix1[ids]=0
    ids = np.reshape(np.where(iy0 < 0), -1)
    if len(ids) > 0:
        iy0[ids] = 0
        iy1[ids] = 0
    ids = np.reshape(np.where(ix1 >= width-1), -1)
    if len(ids) > 0:
        ix0[ids] = width-1
        ix1[ids] = width-1
    ids = np.reshape(np.where(iy1 >= height - 1), -1)
    if len(ids) > 0:
        iy0[ids] = height - 1
        iy1[ids] = height - 1


    ratex = xs - ix0
    ratey = ys - iy0
    if len(img.shape) > 2:
        ratex = ratex.reshape((-1,1))
        ratey = ratey.reshape((-1, 1))

    px0_y0 = img[(iy0,ix0)]
    px0_y1 = img[(iy1,ix0)]
    px1_y0 = img[(iy0,ix1)]
    px1_y1 = img[(iy1,ix1)]

    py0 = px0_y0 * (1.0-ratex) + px1_y0*ratex
    py1 = px0_y1 * (1.0-ratex) + px1_y1*ratex
    p = py0 * (1 - ratey) + py1 * ratey
    return p


def subPix(img, x, y):
    height = img.shape[0]
    width = img.shape[1]
    ix0 = int(x)
    iy0 = int(y)
    ix1 = ix0+1
    iy1 = iy0+1

    if ix0 < 0:
        ix0=ix1=0
    if iy0 < 0:
        iy0=iy1=0
    if ix1 >= width-1:
        ix0=ix1=width-1
    if iy1 >= height-1:
        iy0=iy1=height-1

    ratex = x - ix0
    ratey = y - iy0

    px0_y0 = img[iy0,ix0]
    px0_y1 = img[iy1,ix0]
    px1_y0 = img[iy0,ix1]
    px1_y1 = img[iy1,ix1]

    py0 = px0_y0 * (1.0-ratex) + px1_y0*ratex
    py1 = px0_y1 * (1.0-ratex) + px1_y1*ratex
    p = py0 * (1 - ratey) + py1 * ratey
    return p





def load_pfm(filename):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    file = open(filename,'rb')

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        print 'Incorrect PFM header.'
        exit()

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        print 'Incorrect PFM header.'
        exit()

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)

    reverseIds = range(len(data)-1, -1, -1)

    out = data[reverseIds]


    return out


#Save a Numpy array to a PFM file.
def save_pfm(filename, image, scale = 1):
    file = open(filename,'wb')

    color = None
    image = image.astype(np.float32)
    if image.dtype.name != 'float32':
        print 'Image dtype must be float32.'
        exit()

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        print 'Image must have H x W x 3, H x W x 1 or H x W dimensions.'
        exit()

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    reverseIds = range(len(image)-1, -1, -1)

    out = image[reverseIds]
    out.tofile(file)

def saveAsPly(filename, points, color = (255, 0, 0)):
    color = np.reshape(color, (-1,3))
    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n"%(len(points)))
        f.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n end_header")
        for i,point in enumerate(points):
            if len(color) != len(points):
                f.write("\n%.5f %.5f %.5f %d %d %d"%(point[0], point[1], point[2], color[0][0], color[0][1], color[0][2]))
            else:
                f.write("\n%.5f %.5f %.5f %d %d %d"%(point[0], point[1], point[2], color[i][0], color[i][1], color[i][2]))



