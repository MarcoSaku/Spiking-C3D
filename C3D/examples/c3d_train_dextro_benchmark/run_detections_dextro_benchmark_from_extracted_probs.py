#!/usr/bin/env python

import os
import numpy as np
import math
import json

import sys
sys.path.append("/home/chuck/projects/C3D/python")
import caffe
import array

def get_content_type():
    mapping_file = 'dextro_benchmark_content_type_mapping.json'
    with open(mapping_file, 'r') as fp:
        content_type_mapping = json.load(fp)

    mapping_file = 'dextro_benchmark_content_type_id_mapping.json'
    with open(mapping_file, 'r') as fp:
        content_type_id_mapping = json.load(fp)

    return content_type_mapping, content_type_id_mapping

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

    output_file = 'dextro_benchmark_c3d_performance_from_extracted_probs.csv'
    cwd = os.path.dirname(os.path.realpath(__file__))
    result_path = 'dextro_benchmark_c3d_intermediate_results2_iter_10000_from_extracted_probs'
    ''' c3d_dextro_benchmark2_iter_10000 '''
    if not os.path.isdir(os.path.join(cwd, result_path)):
        os.mkdir(os.path.join(cwd, result_path))

    bufsize = 0
    out = open(output_file, "w", bufsize)

    # read test video list
    test_video_list = 'dextro_benchmark_val_flow_smaller.txt'
    import csv
    reader = csv.reader(open(test_video_list), delimiter=" ")

    # top_N
    top_N = 5
    #top_N = 131 # for debugging

    content_type_mapping, content_type_id_mapping = get_content_type()

    for count, video_and_category in enumerate(reader):
        (video_file, start_frame, category) = video_and_category
        video_name = os.path.splitext(video_file)[0]
        video_basename = video_name.split('/')[-1]
        start_frame = int(start_frame)
        category = int(category)
        if not os.path.isdir(video_name):
            print "[Error] video_name path={} does not exist. Skipping...".format(video_name)
            continue
        video_id = video_name.split('/')[-1]
        feature_file = os.path.join(cwd, 'output', '{0}_{1:05d}'.format(video_id, start_frame) + '.prob')
        print "-"*79
        print "video_name={} ({}-th), start_frame={}, category={}".format(video_name, count+1, start_frame, category)

        avg_pred = read_binary_feature(feature_file)
        print "sum(avg_pred)={}".format(sum(avg_pred))

        result = os.path.join(cwd, result_path, '{0}_frame_{1:05d}_category_{2:04d}_c3d.txt'.format(video_id, start_frame, category))
        np.savetxt(result, avg_pred, delimiter=",")

        sorted_indices = sorted(range(len(avg_pred)), key=lambda k: -avg_pred[k])
        print "-"*5
        for x in range(top_N):
            index = sorted_indices[x]
            prob = round(avg_pred[index]*100,10)
            if category == index:
                hit_or_miss = '!!!!!!!!!!!!!!!  hit !!!!!!!!!!!!!!!'
            else:
                hit_or_miss = ''
            print "[Info] GT:{}, spatial detected:{} (p={}%): {}".format(content_type_mapping[category], content_type_mapping[index], prob, hit_or_miss)

        c3d_rank = sorted_indices.index(category)+1

        out.write("{0}_frame_{1:05d}_category_{2:04d}, {3}\n".format(video_id, start_frame, category, c3d_rank))

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
