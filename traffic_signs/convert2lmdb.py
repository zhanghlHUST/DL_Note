#-*- encoding=utf-8 -*-

import os  
import glob  
import random  
import numpy as np  
  
import cv2  
  
import caffe  
from caffe.proto import caffe_pb2  
import lmdb  
  
#Size of images   输入还是输出 ？
IMAGE_OUT_WIDTH = 32  
IMAGE_OUT_HEIGHT = 32  

# 图像的变换, 直方图均衡化, 以及裁剪到 IMAGE_WIDTH x IMAGE_HEIGHT 的大小  
def transform_img(img, img_width=IMAGE_OUT_WIDTH, img_height=IMAGE_OUT_HEIGHT):  
    #Histogram Equalization  
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])  
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])  
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])  
  
    #Image Resizing, 三次插值  
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)  
    return img  
  
def make_datum(img, label):  
    #image is numpy.ndarray format. BGR instead of RGB
    # ？  
    return caffe_pb2.Datum(  channels=3,  width=IMAGE_OUT_WIDTH, height=IMAGE_OUT_HEIGHT, label=label,  data=np.rollaxis(img, 2).tobytes() )
  
# train_lmdb、validation_lmdb 路径  
train_lmdb = 'train_lmdb'  
validation_lmdb = 'validation_lmdb'  


if os.path.exists( train_lmdb ):
    os.removedirs( train_lmdb )
if os.path.exists( validation_lmdb ):
    os.removedirs( validation_lmdb )

# images data path
train_dir = 'TrainImages'
test_dir = 'TestImages'
train_data = []
# 路径名列表  
for subDir in os.listdir( train_dir ):
     subDirFull = os.path.join( train_dir, subDir)
     for f in os.listdir( subDirFull ):
         if '.ppm' in f:
                train_data.append( subDir+os.path.join(subDirFull, f))

test_data = []
for f in os.listdir( test_dir ):
    if '.ppm' in f:
        test_data.append( os.path.join(test_dir,f))

with open( os.path.join(test_dir,'GT-final_test.csv')) as label_file:
    label_file = list( label_file )
    for i,f in enumerate( test_data ):
        label_line = label_file[i+1]
        label = int( label_line.strip().split(';')[-1] )
        test_data[i] = '%05d'%label + f
        

'''train_data = [img for img in glob.glob("/xxx/*jpg")]  
test_data = [img for img in glob.glob("/xxxx/*jpg")]'''  
  
# Shuffle train_data  
# 打乱数据的顺序  
random.shuffle(train_data)  
    
# 打开 lmdb 环境, 生成一个数据文件，定义最大空间, 1e12 = 1000000000000.0  
in_db = lmdb.open(train_lmdb, map_size=int(1e9))   
with in_db.begin(write=True) as in_txn: # 创建操作数据库句柄  
    for in_idx, img_path in enumerate(train_data):
        print   img_path
        label = int( img_path[0:5] )
        assert img_path[5] == 'T'
        img = cv2.imread(img_path[5:], cv2.IMREAD_COLOR)  
        img = transform_img(img, img_width=IMAGE_OUT_WIDTH, img_height=IMAGE_OUT_HEIGHT)    
        datum = make_datum(img, label)  
        # '{:0>5d}'.format(in_idx):  
        #      lmdb的每一个数据都是由键值对构成的,  
        #      因此生成一个用递增顺序排列的定长唯一的key  
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString()) #调用句柄，写入内存  
        print '{:0>5d}'.format(in_idx) + ':' + img_path[5:]  
  
# 结束后记住释放资源，否则下次用的时候打不开。。。  
in_db.close()   
  
# 创建验证集 lmdb 格式文件  
print '\nCreating validation_lmdb'  
in_db = lmdb.open(validation_lmdb, map_size=int(1e9))  
with in_db.begin(write=True) as in_txn:  
    for in_idx, img_path in enumerate(train_data):  
        label = int( img_path[0:5] )
        assert img_path[5] == 'T'
        img = cv2.imread(img_path[5:], cv2.IMREAD_COLOR)  
        img = transform_img(img, img_width=IMAGE_OUT_WIDTH, img_height=IMAGE_OUT_HEIGHT)  
        datum = make_datum(img, label)  
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())  
        print '{:0>5d}'.format(in_idx) + ':' + img_path[5:]  
in_db.close()  

print '\nFinished processing all images'  
