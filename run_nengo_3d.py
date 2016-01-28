## Caffe format for frames 4,3,16,240,320--->batch_size,channels,num_frames,height,width
## Caffe format for weights (proto.num(), proto.channels(), proto.length(), proto.height(), proto.width())
import cv2
import logging
logging.basicConfig(level=logging.INFO)
#import caffe
import nengo
import nengo_ocl
import numpy as np
#import matplotlib.pyplot as plt
import os
import simplejson
import c3d_caffe
from c3d_caffe.proto import caffe_pb2
from google import protobuf
import time

###Caffe Layer Type Dict ####
layers_type = {33:'data', 30:'conv', 18: 'relu', 31:'pool',14:'fc'};


#### UCF-101 #####
# prototxt_filename = '/home/marcosaviano/C3D-master/examples/c3d_train_ucf101/c3d_feat.prototxt'
# dataset_dir='/opt/Datasets/ucf101/frames/'
# frames_dir='/opt/Datasets/ucf101/frames/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/'
# mean_file='/home/marcosaviano/C3D-master/examples/c3d_train_ucf101/ucf101_train_mean.binaryproto'
# weights_dir='/home/marcosaviano/C3D-master/examples/c3d_train_ucf101/caffe_weights/'

# class Delay(object):
#     def __init__(self, dimensions, timesteps=50):
#         self.history = np.zeros((timesteps, dimensions))
#     def step(self, t, x):
#         self.history = np.roll(self.history, -1)
#         self.history[-1] = x
#         return self.history[0]
#
# delay = Delay(1, timesteps=int(2 / 0.001))

#### YUPENN ####
model_def_file = '/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/proto_files/yupenn_test_poolmean_nobias_deploy.prototxt'
prototxt='/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/proto_files/yupenn_test_poolmean_nobias.prototxt'
model_file = '/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/snapshots/yupenn_train_poolmean_nobias/yupenn_train_poolmean_nobias_iter_5250'
test_video_list = '/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/lst_files/test_yupenn_full_4vid_shuffle.lst'
mean_file='/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/mean_yupenn_full_26vid.binaryproto'

# Neurons variables
#original tau_rc=0.02, tau_ref=0.002,
tau_ref = 0.001
tau_rc = 0.05
alpha = 0.825
amp = 0.063
presentation_time=0.25
num_clip_test=70
crop_size=112
image_resize_dim=(128,171)
get_ind = lambda t: int(t / presentation_time)
###################

def round_array(x, n_values, x_min, x_max):
    if x_min == x_max:
        return

    assert x_min < x_max
    np.clip(x, x_min, x_max, out=x)
    scale = float(n_values - 1) / (x_max - x_min)
    x[:] = np.round(x * scale) / scale


def round_layer(filters,biases, n_values, clip_percent=0):
    #if 'weights' in layer:
    # if filters:
    #for weights in layer['weights']:
    w_min = np.percentile(filters.ravel(), clip_percent)
    w_max = np.percentile(filters.ravel(), 100 - clip_percent)
    round_array(filters, n_values, w_min, w_max)

    # if biases:
    #for biases in layer['biases']:
    b_min = biases.min()
    b_max = biases.max()
    round_array(biases, n_values, b_min, b_max)
    # return filters,biases

def load_data(c3d_net):
    import csv
    reader = csv.reader(open(test_video_list), delimiter=" ")
    labels=np.zeros(num_clip_test)
    blob = c3d_caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_file,'rb').read()
    blob.ParseFromString(data)
    image_mean = np.array(c3d_caffe.io.blobproto_to_array(blob))
    image_mean = np.transpose(np.squeeze(image_mean), (2,3,0,1))

    frames=np.zeros([num_clip_test,3,16,crop_size,crop_size],dtype='float32')
    for count, video_and_category in enumerate(reader):
        if (count)==num_clip_test:
            break
        assert count<num_clip_test
        (video_name, start_frame, category_id) = video_and_category
        video_name = video_name.rstrip('/')
        start_frame = int(start_frame)
        category_id = int(category_id)
        labels[count]=category_id
        # if not os.path.isdir(video_name):
        #     print "[Error] video_name path={} does not exist. Skipping...".format(video_name)
        #     continue
        # video_id = video_name.split('/')[-1]
        # category = video_name.split('/')[-2]

        c3d_depth=c3d_net.blobs['data'].data.shape[2]
        dims = tuple(image_resize_dim) + (3,c3d_depth)
        rgb = np.zeros(shape=dims, dtype=np.float32)
        for i in range(c3d_depth):
                img_file = os.path.join(video_name, '{0:06d}.jpg'.format(start_frame+i))
                img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, dims[1::-1])
                rgb[:,:,:,i] = img

        rgb -= image_mean
        rgb = rgb[8:120, 30:142, :,:]
        rgb= np.transpose(rgb[:,:,:,:], (2,3,0,1))
        frames[count,:,:,:,:]=rgb
    return frames,labels

# Create Node for input frames
def build_layer_data(frames):
    input_3d=nengo.Node(nengo.processes.PresentInput_3D(frames.reshape(frames.shape[0], -1), presentation_time))
    input_3d.label='Input Data'
    return input_3d

# Create layers of network
def build_layer(layers, ii, output_layers,input_shape,type,c3d_net):
    kk=len(output_layers)
    layer=layers[ii]
    target_key=str(layer.name)
    if type=='conv':
        pad=layer.convolution_param.pad
        temporal_pad=layer.convolution_param.temporal_pad
        stride=layer.convolution_param.stride

        filters=c3d_net.layers[ii-1].blobs[0].data
        biases=c3d_net.layers[ii-1].blobs[1].data
        round_layer(filters, biases, 2**8, clip_percent=0)
        convNode = nengo.Node(nengo.processes.Conv3((input_shape), filters, biases))
        nengo.Connection(output_layers[kk-1], convNode)
        new_input_shape=input_shape
        new_input_shape[0]=filters.shape[0]
        convNode.label=target_key
        return convNode,new_input_shape

    elif type=='pool':
        if layer.pooling_param.AVE==1:
            pooltype='avg'
        elif layer.pooling_param.MAX==1:
            pooltype='max'

        kernel_depth=layer.pooling_param.kernel_depth
        kernel_size=layer.pooling_param.kernel_size
        stride=layer.pooling_param.stride
        temporal_stride=layer.pooling_param.temporal_stride
        poolNode = nengo.Node(nengo.processes.Pool3(input_shape, kernel_size, kernel_depth, stride=stride,kind=pooltype,temporal_stride=temporal_stride))
        nengo.Connection(output_layers[kk-1], poolNode, synapse=None)
        poolNode.label=target_key
        c,l,h,w=input_shape
        new_l = (l - 1) / temporal_stride + 1
        new_h = (h - 1) / stride + 1
        new_w = (w - 1) / stride + 1
        new_input_shape=[c,new_l,new_h,new_w]
        return poolNode, new_input_shape

    elif type=='fc':
        filters=c3d_net.layers[ii-1].blobs[0].data
        filters=filters.reshape(filters.shape[-2:])
        biases=c3d_net.layers[ii-1].blobs[1].data
        round_layer(filters, biases, 2**8, clip_percent=0)

        fcNode = nengo.Node(size_in=filters.shape[-2])
        nengo.Connection(output_layers[kk-1], fcNode, transform=filters)

        if biases.max()!=0:
            b = nengo.Node(output=biases)
            nengo.Connection(b, fcNode,synapse=None)
            b.label=target_key+'_biases'
        fcNode.label=target_key

        return fcNode, None

    elif type=='drop':
        dropNode = nengo.Node(size_in=output_layers[kk-1].size_out)
        nengo.Connection(output_layers[kk-1], dropNode, transform=layer.dropout_param.dropout_ratio)
        dropNode.label=target_key
        return dropNode, input_shape

    elif type=='relu':
        fcNeurons = nengo.Ensemble(output_layers[kk-1].size_out, dimensions=1)
        #nengo.Connection(output_layers[kk-1], fcNeurons.neurons,synapse=None)
        nengo.Connection(output_layers[kk-1], fcNeurons.neurons)
        fcNeurons.neuron_type = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
        fcNeurons.gain = alpha * np.ones(output_layers[kk-1].size_out)
        fcNeurons.bias = 1 * np.ones(output_layers[kk-1].size_out)
        u2 = nengo.Node(size_in=output_layers[kk-1].size_out)
        #nengo.Connection(fcNeurons.neurons, u2, synapse=None)
        nengo.Connection(fcNeurons.neurons, u2, transform=amp, synapse=None)
        u2.label=target_key
        return u2, input_shape

def write_files(y,dt,y_filt):
    import simplejson
    print 'Writing file...'
    f = open('tests/y70_1.txt', 'w')
    f_labels=open('tests/labels.txt', 'w')
    simplejson.dump(y.tolist(), f)
    simplejson.dump(labels.tolist(), f_labels)
    f.close()
    f_labels.close()
    import simplejson
    f = open('tests/y_filt70_1.txt', 'w')
    simplejson.dump(y_filt.tolist(), f)
    f.close()
    print 'Files written!'

def my_error_new(dt, labels, t, y,y_filt):
    print 'Samples classification'
    ct = 0.005  # classification time
    # take average class over last 5 ms of each presentation
    pn = int(presentation_time / dt)
    cn = int(ct / dt)
    n = y_filt.shape[0] / pn
    assert cn <= pn
    probs_ct=y_filt.reshape(n, pn, y_filt.shape[1])[:, -cn:, :]
    probs = probs_ct.mean(1)

    labels = labels[:n]
    assert probs.shape[0] == labels.shape[0]
    inds = np.argsort(probs, axis=1)
    y_pred=inds[:, -1]
    top1errors = y_pred != labels
    top5errors = np.all(inds[:, -5:] != labels[:, None], axis=1)
    for ii in range(0,n):
        if top1errors[ii]==True:
            print str(ii)+" label:"+str(int(labels[ii]))+" y_pred:"+str(y_pred[ii])+'\tmiss!!'
        else:
            print str(ii)+" label:"+str(int(labels[ii]))+" y_pred:"+str(y_pred[ii])+'\tHIT!!'
    tp=np.sum(y_pred==labels)
    print "Correct samples classified: "+str(tp)+"/"+str(len(labels))
    print "True Positive Rate: "+str(1-top1errors.mean())+"%"

    return y_pred, top1errors,

#num_batches,num_frames,num_channel,height,width
c3d_net = c3d_caffe.Net(model_def_file, model_file)
print 'Presentation time per clip: '+str(presentation_time)
print "Loading data: "+str(num_clip_test)+" clips..."
frames,labels=load_data(c3d_net)

# Read Network Structure from prototxt file
print 'Reading Network Structure...'
net = caffe_pb2.NetParameter()
with open(prototxt) as f:
    protobuf.text_format.Parse(f.read(), net)
layers_name=[]
for ii in range(0,len(net.layers)):
    layers_name.append(str(net.layers[ii].name))

print "Creating Network..."
network = nengo.Network('SNN')
network.config[nengo.Connection].synapse = nengo.synapses.Alpha(0.005)
#create Network
output_layers = []
target_key=None
input_shape=[3,16,crop_size,crop_size]

with network:
    #for ii in range(0, 2):
    for ii in range(0, len(net.layers)):
        type_id=net.layers[ii].type
        type=layers_type.get(type_id)
        if type=='conv' or type=="pool" or type=="fc" or type=="drop" or type=="relu":
        #if type=='conv' or type=="pool" or type=="fc" or type=="drop":
            out,input_shape=build_layer(net.layers,ii,output_layers,input_shape,type,c3d_net)
            output_layers.append(out)

        elif type=="data":
            out=build_layer_data(frames)
            output_layers.append(out)
    yp = nengo.Probe(output_layers[-1], synapse=None)
    #in_probe=nengo.Probe(output_layers[0], synapse=None)
start_time = time.time()
print("Creating Simulator...")
sim = nengo_ocl.Simulator(network)
print("--- Simulator Created in %s seconds ---" % (time.time() - start_time))

# sim = nengo_ocl.Simulator(network, profiling=True)

print("Starting Simulation...")
sim.run(num_clip_test * presentation_time)
print 'Getting probe...'
dt = sim.dt
t = sim.trange()
y = sim.data[yp]
#xx=sim.data[in_probe]
print 'Probe ok!'
s = nengo.synapses.Alpha(0.005)
y_filt = nengo.synapses.filtfilt(y, s, dt)
write_files(y,dt,y_filt)
#y_pred,my_errors=my_error(dt, labels, t, y)
y_pred,my_errors=my_error_new(dt, labels, t, y, y_filt)
print "Done!"


