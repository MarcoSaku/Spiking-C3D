#!/usr/bin/env python

import os
import numpy as np
import math
import json
import array

import sys
sys.path.append("/home/chuck/projects/C3D/python")
import caffe

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

def read_binary_feature(file):
    f = open(file, "rb") # read binary data
    s = f.read() # read all bytes into a string
    f.close()

    (n, c, l, h, w) = array.array("i", s[:20])
    feature = np.array(array.array("f", s[20:]))

    # sanity check: checks!
    #print "n={},c={},l={},h={},w={}".format(n,c,l,h,w)
    #sum_feature = np.sum(feature)
    #print "sum_feature={}".format(sum_feature)
    #print "feature={}".format(np.squeeze(feature))

    return feature

def main():

    ucf_categories = get_ucf_categories()
    output_file = 'ucf101_c3d_performance_from_probs.csv'
    cwd = os.path.dirname(os.path.realpath(__file__))

    bufsize = 0
    out = open(output_file, "w", bufsize)

    # read test video list
    #test_video_list = '../c3d_finetuning/test_01.lst'
    #test_video_list = './test_54pct_accuracy.lst'
    test_video_list = './test_100pct_accuracy.lst'
    import csv
    reader = csv.reader(open(test_video_list), delimiter=" ")

    # top_N
    top_N = 5

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
        feature_file = os.path.join(cwd, 'output', category, 'v_'+video_id, '{0:05d}'.format(start_frame) + '.prob')

        print "-"*79
        print "video_name={} ({}-th), video_id={}, start_frame={}, category={}, category_id={}, feature_file={}".format(video_name, count+1, video_id, start_frame, category, category_id, feature_file)

        avg_pred = read_binary_feature(feature_file)
        #print "avg_pred.shape={}".format(avg_pred.shape)
        print "sum(avg_pred)={}".format(sum(avg_pred))

        sorted_indices = sorted(range(len(avg_pred)), key=lambda k: -avg_pred[k])
        print "-"*5
        for x in range(top_N):
            index = sorted_indices[x]
            prob = round(avg_pred[index]*100,10)
            if category.lower() == ucf_categories[index].lower():
                hit_or_miss = '!!!!!!!!!!!!!!!  hit !!!!!!!!!!!!!!!'
            else:
                hit_or_miss = ''
            print "[Info] GT:{}, c3d detected:{} (p={}%): {}".format(category, ucf_categories[index], prob, hit_or_miss)

        c3d_rank = sorted_indices.index(category_id)+1

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
    use_image: false
    mean_file: "train01_16_128_171_mean.binaryproto"
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
    shuffle: true
  }
}
'''
