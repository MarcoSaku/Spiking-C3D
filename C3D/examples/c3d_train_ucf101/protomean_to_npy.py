import sys
sys.path.append("/home/chuck/projects/C3D/python")
import caffe
import numpy as np
import os

protomean = 'ucf101_train_mean.binaryproto'
npymean = 'ucf101_train_mean.npy'

if not os.path.isfile(npymean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( protomean, 'rb' ).read()
    blob.ParseFromString(data)
    blob.num = 16
    blob.channels = 3
    blob.height = 128
    blob.width = 171
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    print "arr.shape={}".format(arr.shape)
    #out = arr[0]
    #np.save( npymean , out )
    np.save( npymean, arr )
