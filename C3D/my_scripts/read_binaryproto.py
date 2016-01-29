import caffe
import numpy as np
import sys

# if len(sys.argv) != 3:
# print "Usage: python convert_protomean.py proto.mean out.npy"
# sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( 'ucf101_train_mean.binaryproto', 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]