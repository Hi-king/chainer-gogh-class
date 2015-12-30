import chainer
from models import *
import os
import pickle
from PIL import Image
import numpy as np

def subtract_mean(x0):
    x = x0.copy()
    x[0,0,:,:] -= 120
    x[0,1,:,:] -= 120
    x[0,2,:,:] -= 120
    return x

def add_mean(x0):
    x = x0.copy()
    x[0,0,:,:] += 120
    x[0,1,:,:] += 120
    x[0,2,:,:] += 120
    return x

def save_image(img, width, out_dir, it):
    def to_img(x):
        _, new_h, new_w = x.shape
        im = np.zeros((new_h,new_w,3))
        im[:,:,0] = x[2,:,:]
        im[:,:,1] = x[1,:,:]
        im[:,:,2] = x[0,:,:]
        def clip(a):
            return 0 if a<0 else (255 if a>255 else a)
        im = np.vectorize(clip)(im).astype(np.uint8)
        Image.fromarray(im).save(out_dir+"/im_%05d.png"%it)

    img_cpu = add_mean(img)
    to_img(img_cpu[0])
 
class Clip(chainer.Function):
    def forward(self, x):
        x = x[0]
        ret = cuda.elementwise(
            'T x','T ret',
            '''
                ret = x<-120?-120:(x>136?136:x);
            ''','clip')(x)
        return ret

def get_matrix(y):
    ch = y.data.shape[1]
    wd = y.data.shape[2]
    gogh_y = F.reshape(y, (ch,wd**2))
    gogh_matrix = F.matmul(gogh_y, gogh_y, transb=True)/np.float32(ch*wd**2)
    return gogh_matrix

def image_crop(img_file, width, xp):
    gogh = Image.open(img_file)
    orig_w, orig_h = gogh.size[0], gogh.size[1]
    if orig_w>orig_h:
        new_w = width*orig_w/orig_h
        new_h = width
        gogh = np.asarray(gogh.resize((new_w,new_h)))[:,:,:3].transpose(2, 0, 1)[::-1].astype(np.float32)
        gogh = gogh.reshape((1,3,new_h,new_w))
        left = (new_w - width)/2
        gogh = gogh[:,:,:,left:left+width]
        gogh = subtract_mean(gogh)
    else:
        new_w = width
        new_h = width*orig_h/orig_w
        gogh = np.asarray(gogh.resize((new_w,new_h)))[:,:,:3].transpose(2, 0, 1)[::-1].astype(np.float32)
        gogh = gogh.reshape((1,3,new_h,new_w))
        top = (new_h - width)/2
        gogh = gogh[:,:,top:top+width,:]
        gogh = subtract_mean(gogh)

    return xp.asarray(gogh), new_w, new_h

def load_nn(modelname):
    cachepath = "{}.dump".format(modelname)
    if os.path.exists(cachepath):
        nn = pickle.load(open(cachepath))
        return nn
        
    if 'nin' in modelname:
        nn = NIN()
    elif 'vgg' in modelname:
        nn = VGG()
    elif 'i2v' in modelname:
        nn = I2V()
    elif 'googlenet' in modelname:
        nn = GoogLeNet()
    else:
        print 'invalid model name. you can use (nin, vgg, i2v, googlenet)'
        exit(1)

    with open(cachepath, "w+") as f:
        pickle.dump(nn, f, pickle.HIGHEST_PROTOCOL)
    return nn
