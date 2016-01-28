#!/usr/bin/env python

import os
import numpy as np
import math
import json

import sys
sys.path.append("/home/marcosaviano/python-C3D/C3D/python")
import c3d_caffe

from c3d_caffe.c3d_classify import c3d_classify

def get_ucf_categories():
    category = [
        'ApplyEyeMakeup',
        'ApplyLipstick',
        'Archery',
        'BabyCrawling',
        'BalanceBeam',
        'BandMarching',
        'BaseballPitch',
        'Basketball',
        'BasketballDunk',
        'BenchPress',
        'Biking',
        'Billiards',
        'BlowDryHair',
        'BlowingCandles',
        'BodyWeightSquats',
        'Bowling',
        'BoxingPunchingBag',
        'BoxingSpeedBag',
        'BreastStroke',
        'BrushingTeeth',
        'CleanAndJerk',
        'CliffDiving',
        'CricketBowling',
        'CricketShot',
        'CuttingInKitchen',
        'Diving',
        'Drumming',
        'Fencing',
        'FieldHockeyPenalty',
        'FloorGymnastics',
        'FrisbeeCatch',
        'FrontCrawl',
        'GolfSwing',
        'Haircut',
        'Hammering',
        'HammerThrow',
        'HandstandPushups',
        'HandstandWalking',
        'HeadMassage',
        'HighJump',
        'HorseRace',
        'HorseRiding',
        'HulaHoop',
        'IceDancing',
        'JavelinThrow',
        'JugglingBalls',
        'JumpingJack',
        'JumpRope',
        'Kayaking',
        'Knitting',
        'LongJump',
        'Lunges',
        'MilitaryParade',
        'Mixing',
        'MoppingFloor',
        'Nunchucks',
        'ParallelBars',
        'PizzaTossing',
        'PlayingCello',
        'PlayingDaf',
        'PlayingDhol',
        'PlayingFlute',
        'PlayingGuitar',
        'PlayingPiano',
        'PlayingSitar',
        'PlayingTabla',
        'PlayingViolin',
        'PoleVault',
        'PommelHorse',
        'PullUps',
        'Punch',
        'PushUps',
        'Rafting',
        'RockClimbingIndoor',
        'RopeClimbing',
        'Rowing',
        'SalsaSpin',
        'ShavingBeard',
        'Shotput',
        'SkateBoarding',
        'Skiing',
        'Skijet',
        'SkyDiving',
        'SoccerJuggling',
        'SoccerPenalty',
        'StillRings',
        'SumoWrestling',
        'Surfing',
        'Swing',
        'TableTennisShot',
        'TaiChi',
        'TennisSwing',
        'ThrowDiscus',
        'TrampolineJumping',
        'Typing',
        'UnevenBars',
        'VolleyballSpiking',
        'WalkingWithDog',
        'WallPushups',
        'WritingOnBoard',
        'YoYo'
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

    ucf_categories = get_ucf_categories()
    output_file = 'ucf101_c3d_performance_using_caffe.csv'
    cwd = os.path.dirname(os.path.realpath(__file__))
    result_path = 'ucf101_c3d_intermediate_results'

    bufsize = 0
    out = open(output_file, "w", bufsize)

    # model
    model_def_file = 'conv3d_ucf101_deploy.prototxt'
    model_file = 'conv3d_ucf101_iter_50000'
    #mean_file = 'ucf101_train_mean.npy'
    mean_file = './ucf101_train_mean.binaryproto'
    net = caffe.Net(model_def_file, model_file)

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
    #test_video_list = '../c3d_finetuning/test_01.lst'
    #test_video_list = './test_100pct_accuracy.lst'
    test_video_list = './test_54pct_accuracy.lst'
    import csv
    reader = csv.reader(open(test_video_list), delimiter=" ")

    # top_N
    top_N = 3

    # network param
    prob_layer = 'prob'

    for count, video_and_category in enumerate(reader):
        (video_name, start_frame, category_id) = video_and_category
        video_name = video_name.rstrip('/')
        start_frame = int(start_frame)
        category_id = int(category_id)
        if not os.path.isdir(video_name):
            print "[Error] video_name path={} does not exist. Skipping...".format(video_name)
            continue
        video_id = video_name.split('/')[-1][2:]
        category = video_name.split('/')[-2]

        print "-"*79
        print "video_name={} ({}-th), video_id={}, start_frame={}, category={}, category_id={}".format(video_name, count+1, video_id, start_frame, category, category_id)

        result = os.path.join(cwd, result_path, '{0}_frame_{1:05d}_c3d.txt'.format(video_id, start_frame))
        if os.path.isfile(result) and not force_save:
            print "[Info] intermediate output file={} has been already saved. Skipping...".format(result)
            avg_pred = np.loadtxt(result)
        else:
            blob = caffe.proto.caffe_pb2.BlobProto()
            data = open(mean_file,'rb').read()
            blob.ParseFromString(data)
            image_mean = np.array(caffe.io.blobproto_to_array(blob))
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
                hit_or_miss = '!!!!!!!!!!!!!!!  hit !!!!!!!!!!!!!!!'
            else:
                hit_or_miss = ''
            print "[Info] GT:{}, c3d detected:{} (p={}%): {}".format(category, ucf_categories[index], prob, hit_or_miss)
        c3d_rank = sorted_indices.index(category_id) + 1
        out.write("{0}_{1:05d}, {2}\n".format(video_id, start_frame, c3d_rank))
    out.close()

if __name__ == "__main__":
    main()

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
