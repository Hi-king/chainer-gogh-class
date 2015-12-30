import argparse
import os
import sys

import numpy as np

import chainer
import chainer.links
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe
from chainer import Variable, optimizers

import collections
import pickle
import random
import chainer_gogh_lib

parser = argparse.ArgumentParser(
    description='A Neural Algorithm of Artistic Style')
parser.add_argument('--model', '-m', default='nin',
                    help='model file (nin, vgg, i2v, googlenet)')
parser.add_argument('orig_img', default='orig.png')
parser.add_argument('style_img', default='style.png')
parser.add_argument('--out_dir', '-o', default='output',
                    help='Output directory')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--iter', default=5000, type=int,
                    help='number of iteration')
parser.add_argument('--lr', default=4.0, type=float,
                    help='learning rate')
parser.add_argument('--lam', default=0.005, type=float,
                    help='original image weight / style weight ratio')
parser.add_argument('--content_width', default=435, type=int)
parser.add_argument('--style_width', default=435, type=int)
parser.add_argument('--style_split', default=1, type=int)
parser.add_argument('--tag_names_file')

class TagName(object):
    def __init__(self, filename):
        self.tag_names = [line.rstrip() for line in open(filename)]
        self.top_n = 5
        
    def show_tagnames(self, probabilities, target_ids=[]):
        top_names = []
        for i, tag_id in enumerate(np.argsort(-probabilities, axis=0)):
            print(tag_id, probabilities[tag_id], self.tag_names[tag_id])
            if i > self.top_n:
                break

        for tag_id in target_ids:
            print(tag_id, probabilities[tag_id], self.tag_names[tag_id])
        return top_names

if __name__ == '__main__':
    args = parser.parse_args()
    try:
        os.mkdir(args.out_dir)
    except:
        pass
    
    if args.gpu >= 0:
        cuda.check_cuda_available()
        chainer.Function.type_check_enable = False
        cuda.get_device(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = np
    
    nn = chainer_gogh_lib.load_nn(args.model)
    if args.gpu>=0:
    	nn.model.to_gpu()
    
    img_content,nw,nh = chainer_gogh_lib.image_crop(args.orig_img, 224, xp)
    img_style,_,_ = chainer_gogh_lib.image_crop(args.style_img, args.style_width, xp)
    img_gen = xp.random.uniform(-20, 20, img_content.shape, dtype=np.float32)
    img_gen = img_content.copy()
    if not args.tag_names_file is None:
        tag_name_repository = TagName(args.tag_names_file)

    ## VGG 1 class
    #original_predict = nn.predict(Variable(img_content, volatile=True)).data.get()
    #original_predict = xp.zeros(original_predict.shape, dtype=original_predict.dtype)
    TARGET_PREDCIT = 153 # Maltese dog, Maltese terrier, Maltese 
    #TARGET_PREDCIT = 282 # tiger cat
    #original_predict[0, 328] = 1
    #original_predict[0, 390] = 1
    #original_predict[0, 599] = 1
    #original_predict[0, 607] = 1
    #original_predict[0, 282] = 1

    ## il2vec sigmoid
    original_predict_sigmoid = F.sigmoid(nn.predict(Variable(img_content, volatile=True))).data.get()[0]
    original_predict_sigmoid[5] = 1 # breasts
    original_predict_sigmoid[39] = 1 # large breasts
    original_predict_sigmoid[130] = 1 # huge breasts
    # original_predict_sigmoid[373] = 1 #small breasts
    # original_predict_sigmoid[1536] = 1 #safe
    # original_predict_sigmoid[1537] = 0 #questionable
    # original_predict_sigmoid[1538] = 0 #explicit

    PRINT_IDS = []

    #print(np.argsort(-original_predict.data.get(), axis=1)[:, :10])
    #style_mats = [chainer_gogh_lib.get_matrix(y) for y in nn.forward(Variable(img_style, volatile=True))]
    style_mats = [chainer_gogh_lib.get_matrix(y) for y in nn.forward(Variable(img_content, volatile=True))]

    ## setup
    img_gen = chainer.links.Parameter(img_gen)
    optimizer = optimizers.Adam(alpha=args.lr)
    optimizer.setup(img_gen)

    for i in xrange(args.iter):
        img_gen.zerograds()
        x = img_gen.W

        ## predict loss
        predict = nn.predict(x)
        L_predict = F.softmax_cross_entropy(
            predict, 
            Variable(xp.array([TARGET_PREDCIT]))
        )
        # L_predict = F.mean_squared_error(
        #     predict, 
        #     Variable(original_predict)
        # )
        # L_predict = F.sigmoid_cross_entropy(
        #     F.sigmoid(predict),
        #     Variable(xp.array(
        #         [original_predict_indexs])
        # ))
        # L_predict = F.mean_squared_error(
        #     F.sigmoid(predict), 
        #     Variable(xp.array([original_predict_sigmoid]))
        # ) * 1

        ## style loss
        L_style = Variable(xp.zeros((), dtype=np.float32))
        y = nn.forward(x)
        for layer_index in range(len(y)):
            ch = y[layer_index].data.shape[1]
            wd = y[layer_index].data.shape[2]
            gogh_y = F.reshape(y[layer_index], (ch,wd**2))
            gogh_matrix = F.matmul(gogh_y, gogh_y, transb=True)/np.float32(ch*wd**2)
            #gogh_matrix = chainer_gogh_lib.get_matrix(y[layer_index])

            l = np.float32(nn.beta[layer_index])*F.mean_squared_error(gogh_matrix, Variable(style_mats[layer_index].data))/np.float32(len(y))
            L_style += l

        #L = L_predict*1000 + L_style #VGG
        L = L_predict*10000 + L_style #i2v
        # L = L_predict
        # L = L_style
        L.backward()
	img_gen.W.grad = x.grad
        optimizer.update()


        if i%10 == 0:
            print(L_predict.data, L_style.data)
            if not args.tag_names_file is None:
                tag_name_repository.show_tagnames(predict.data.get()[0], target_ids=PRINT_IDS)

        if i%10==0:
            print(img_gen.W.data.shape)
            if args.gpu >= 0:
                chainer_gogh_lib.save_image(img_gen.W.data.get(), args.content_width, args.out_dir, i)
            else:
                chainer_gogh_lib.save_image(img_gen.W.data, args.content_width, args.out_dir, i)
