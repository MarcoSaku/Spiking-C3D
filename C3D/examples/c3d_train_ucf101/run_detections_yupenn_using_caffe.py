#!/usr/bin/env python

import os
import numpy as np
import math
import json
import cv2

import sys
sys.path.append("/home/marcosaviano/python-C3D/C3D/python")
import c3d_caffe

def c3d_classify(
        vid_name,
        image_mean,
        net,
        start_frame,
        image_resize_dim,
        prob_layer='prob',
        multi_crop=False
        ):
    ''' start_frame is 1-based and the first image file is image_0001.jpg '''

    # infer net params
    batch_size = net.blobs['data'].data.shape[0]
    c3d_depth = net.blobs['data'].data.shape[2]
    num_categories = net.blobs['prob'].data.shape[1]

    # selection
    dims = tuple(image_resize_dim) + (3,c3d_depth)
    rgb = np.zeros(shape=dims, dtype=np.float32)
    rgb_flip = np.zeros(shape=dims, dtype=np.float32)

    for i in range(c3d_depth):
        img_file = os.path.join(vid_name, '{0:06d}.jpg'.format(start_frame+i))
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, dims[1::-1])
        rgb[:,:,:,i] = img
        rgb_flip[:,:,:,i] = img[:,::-1,:]

    # substract mean
    image_mean = np.transpose(np.squeeze(image_mean), (2,3,0,1))
    rgb -= image_mean
    rgb_flip -= image_mean[:,::-1,:,:]

    if multi_crop:
        # crop (112-by-112)
        rgb_1 = rgb[:112, :112, :,:]
        rgb_2 = rgb[:112, -112:, :,:]
        rgb_3 = rgb[8:120, 30:142, :,:]
        rgb_4 = rgb[-112:, :112, :,:]
        rgb_5 = rgb[-112:, -112:, :,:]
        rgb_f_1 = rgb_flip[:112, :112, :,:]
        rgb_f_2 = rgb_flip[:112, -112:, :,:]
        rgb_f_3 = rgb_flip[8:120, 30:142, :,:]
        rgb_f_4 = rgb_flip[-112:, :112, :,:]
        rgb_f_5 = rgb_flip[-112:, -112:, :,:]

        rgb = np.concatenate((rgb_1[...,np.newaxis],
                              rgb_2[...,np.newaxis],
                              rgb_3[...,np.newaxis],
                              rgb_4[...,np.newaxis],
                              rgb_5[...,np.newaxis],
                              rgb_f_1[...,np.newaxis],
                              rgb_f_2[...,np.newaxis],
                              rgb_f_3[...,np.newaxis],
                              rgb_f_4[...,np.newaxis],
                              rgb_f_5[...,np.newaxis]), axis=4)
    else:
        rgb_3 = rgb[8:120, 30:142, :,:]
        #rgb_f_3 = rgb_flip[8:120, 30:142, :,:]
        #rgb = np.concatenate((rgb_3[...,np.newaxis],
        #                      rgb_f_3[...,np.newaxis]), axis=4)
        rgb = rgb_3[...,np.newaxis]

    prediction = np.zeros((num_categories,rgb.shape[4]))

    if rgb.shape[4] < batch_size:
        net.blobs['data'].data[:rgb.shape[4],:,:,:,:] = np.transpose(rgb, (4,2,3,0,1))
        output = net.forward()
        prediction = np.transpose(np.squeeze(output[prob_layer][:rgb.shape[4],:,:,:,:], axis=(2,3,4)))
    else:
        num_batches = int(math.ceil(float(rgb.shape[4])/batch_size))
        for bb in range(num_batches):
            span = range(batch_size*bb, min(rgb.shape[4],batch_size*(bb+1)))
            net.blobs['data'].data[...] = np.transpose(rgb[:,:,:,:,span], (4,2,3,0,1))
            output = net.forward(blobs=['data','conv1a'])
            prediction[:, span] = np.transpose(np.squeeze(output[prob_layer], axis=(2,3,4)))

    return prediction



def get_yupenn_categories():
    category = [
        'Beach',
        'Elevator',
        'ForestFire',
        'Fountain',
        'Highway',
        'LightningStorm',
        'Ocean',
        'Railway',
        'RushingRiver',
        'SkyClouds',
        'Snowing',
        'Street',
        'Waterfall',
        'WindmillFarm'
        ]

    return category

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z

def main():

    # force save
    force_save = False
    num_clip_test=50
    yupenn_categories = get_yupenn_categories()
    output_file = 'yupenn_c3d_performance_using_caffe.csv'
    cwd = os.path.dirname(os.path.realpath(__file__))
    result_path = 'yupenn_c3d_intermediate_results'

    bufsize = 0
    out = open(output_file, "w", bufsize)

    # model
    model_def_file = 'conv3d_yupenn_deploy.prototxt'
    model_file = '/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/snapshots/yupenn_train_poolmean_nobias/yupenn_train_poolmean_nobias_iter_5250'
    #mean_file = 'ucf101_train_mean.npy'
    mean_file = '/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/mean_yupenn_full_26vid.binaryproto'
    net = c3d_caffe.Net(model_def_file, model_file)

    # caffe init
    gpu_id = 0
    net.set_device(gpu_id)
    net.set_mode_gpu()
    #net.set_mode_cpu()
    net.set_phase_test()
    #net.set_mean('data', '../python/caffe/imagenet/ilsvrc_2012_mean.npy')
    #net.set_channel_swap('data', (2,1,0))
    #net.set_input_scale('data', 255.0)

    # read test video list
    test_video_list = '/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/lst_files/test_yupenn_full_4vid_shuffle.lst'
    import csv
    reader = csv.reader(open(test_video_list), delimiter=" ")

    # top_N
    top_N = 1

    # network param
    prob_layer = 'prob'
    tp=0
    for count, video_and_category in enumerate(reader):
        if (count)==num_clip_test:
            break
        (video_name, start_frame, category_id) = video_and_category
        video_name = video_name.rstrip('/')
        start_frame = int(start_frame)
        category_id = int(category_id)
        if not os.path.isdir(video_name):
            print "[Error] video_name path={} does not exist. Skipping...".format(video_name)
            continue
        #video_id = video_name.split('/')[-1][2:]
        video_id = video_name.split('/')[-1]
        category = video_name.split('/')[-2]

        print "-"*79
        print "video_name={} ({}-th), video_id={}, start_frame={}, category={}, category_id={}".format(video_name, count, video_id, start_frame, category, category_id)

        result = os.path.join(cwd, result_path, '{0}_frame_{1:05d}_c3d.txt'.format(video_id, start_frame))
        # if os.path.isfile(result) and not force_save:
        #     print "[Info] intermediate output file={} has been already saved. Skipping...".format(result)
        #     avg_pred = np.loadtxt(result)
        # else:
        blob = c3d_caffe.proto.caffe_pb2.BlobProto()
        data = open(mean_file,'rb').read()
        blob.ParseFromString(data)
        image_mean = np.array(c3d_caffe.io.blobproto_to_array(blob))
        prediction = c3d_classify(
                vid_name=video_name,
                image_mean=image_mean,
                net=net,
                start_frame=start_frame,
                image_resize_dim = (128,171),
                prob_layer=prob_layer,
                multi_crop=False
                )
        if prediction.ndim == 2:
            avg_pred = np.mean(prediction, axis=1)
        else:
            avg_pred = prediction
        #print "prediction.shape={}, avg_pred.shape={}".format(prediction.shape, avg_pred.shape)
        #avg_pred_fc8 = np.mean(prediction, axis=1)
        #avg_pred = softmax(avg_pred_fc8)
        np.savetxt(result, avg_pred, delimiter=",")
        sorted_indices = sorted(range(len(avg_pred)), key=lambda k: -avg_pred[k])
        print "-"*5
        for x in range(top_N):
            index = sorted_indices[x]
            prob = round(avg_pred[index]*100,10)
            if category_id == index:
                tp +=1
                hit_or_miss = '!!!!!!!!!!!!!!!  hit !!!!!!!!!!!!!!!'
            else:
                hit_or_miss = ''
            print "[Info] GT:{}, c3d detected:{} (p={}%): {}".format(category, yupenn_categories[index], prob, hit_or_miss)
        c3d_rank = sorted_indices.index(category_id) + 1
        out.write("{0}_{1:05d}, {2}\n".format(video_id, start_frame, c3d_rank))
    print "Correct classifications:"+str(tp)+"/"+str(num_clip_test)
    print "True Positive Rate: "+str(tp/(num_clip_test*1.0))+"%"
    out.close()

if __name__ == "__main__":
    main()
    print 'Done!'

'''
layers {
  name: "data"
  type: VIDEO_DATA
  top: "data"
  top: "label"
  image_data_param {
    source: "dextro_benchmark_val_flow_smaller.txt"
    use_image: true
    mean_file: "ucf101_train_mean.binaryproto"
    use_temporal_jitter: false
    #batch_size: 30
    #batch_size: 2
    batch_size: 2
    crop_size: 112
    mirror: false
    show_data: 0
    new_height: 128
    new_width: 171
    new_length: 16
    shuffle: false
  }
}
'''
