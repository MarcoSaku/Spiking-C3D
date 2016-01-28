#!/usr/bin/env python

import csv
import sys
import os
import cv2
import glob

from common.env import environ
from common.s3_interface import S3Interface
from boto.s3.key import Key
import boto.exception as boto_exception

# a directory that saves videos temporarily
tmp_video_download_dir = '/media/TB/Videos/periscope-dog-and-cat'

# download existing video files from s3?
force_download_from_s3 = False

# read a video list
video_list_file = 'dog_cat_streams.csv'

# intermediate files
input_file = '/home/chuck/projects/C3D/examples/c3d_feature_extraction/prototxt/periscope_dog_cat_input_list_video_scene_segmented.txt'
output_file = '/home/chuck/projects/C3D/examples/c3d_feature_extraction/prototxt/periscope_dog_cat_output_list_video_prefix_scene_segmented.txt'
outdir='/home/chuck/c3d_features'

# use scene-segmentation
use_scene = True

if not os.path.isdir(outdir):
    os.makedirs(outdir)

videos = []
f = open(video_list_file, 'rt')
try:
    reader = csv.reader(f)
    for row in reader:
        video_url = row[0]
        category = int(row[1])
        #print "[Info] video_url='{}', category={}...".format(video_url, category)
        videos.append((video_url, category))
finally:
    f.close()

in_file_obj = open(input_file, "a")
out_file_obj = open(output_file, "a")

# for each video...
for video in videos:
    video_url, category = video
    print "[Info] Procedssing video_url='{}', category={}...".format(video_url, category)
    if category == 9:
        category_zero_or_one = 0
    elif category == 21:
        category_zero_or_one = 1
    else:
        category_zero_or_one = -1
        print "[Warning] content_type is neight 9 nor 21! Continuing..."

    # - download from s3
    # `s3://periscope-demo/concatenated_replays/<id>.mp4`
    video_id = video_url.split('/')[-1]
    local_video_file = os.path.join(tmp_video_download_dir, video_id + '.mp4')
    #print "[Info] local_video_file={}".format(local_video_file)
    if os.path.isfile(local_video_file) or not force_download_from_s3:
        print "- [Info] This video has been already locally saved. Skipping..."
    else:
        videos_s3_folder_path = 'concatenated_replays'
        video_s3_filename = "{0}/{1}".format(videos_s3_folder_path, video_id + '.mp4')
        s3_interface = S3Interface('periscope-demo')
        video_key = Key(s3_interface.bucket, video_s3_filename)
        if not video_key.exists():
            print "- [Error] s3 key does not exist. Skipping..."
            continue
        try:
            video_key.get_contents_to_filename(local_video_file)
        except boto_exception.S3ResponseError:
            print "- [Error] Can not download s3 object. Skipping..."
            continue
        finally:
            video_key.close()

    # - get number of frames
    cap = cv2.VideoCapture(local_video_file)

    if not cap.isOpened():
        print "- [Error] Could not open video file={}. Skipping...".format(local_video_file)
        continue

    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    print "- [Info] num_frames={}".format(num_frames)
    outdir_plus_video_id = os.path.join(outdir, video_id)

    frame_increment = 100
    c3d_frame_size = 16

    if use_scene:
        # get scene segmentation file
        scene_file = glob.glob(os.path.join(outdir, '*_' + video_id + '_scene.txt'))
        if len(scene_file) == 1:
            print "scene_file found = {}".format(scene_file)
        else:
            #print "scene_file cannot be found w/ len(scene_file)={}, globbed by {}".format(len(scene_file), os.path.join(outdir, '*_' + video_id + '_scene.txt'))
            continue

        # check if first segment has at least 200 frames
        fp = open(scene_file[0])
        first_seg = ''
        for linenum, line in enumerate(fp):
            if linenum == 2:
                first_seg = line.strip()
            elif linenum > 2:
                break
        fp.close()

        if len(first_seg) > 0 and first_seg[0] == '1' :
            last_frame = int(float(first_seg.split(',')[2]))
            #if num_frames > 300:
            #    frame_start = 100
            #    last_frame = last_frame - 100 - c3d_frame_size
            if num_frames > 500 + c3d_frame_size:
                frame_start = 100
                last_frame = 600
            else:
                continue
        else:
            continue
    else:
        frame_start = 0
        last_frame = num_frames-c3d_frame_size+1

    for frame_num in range(frame_start, last_frame, frame_increment):
        print "--- [Info] Processing frame_num={}...".format(frame_num)

        # - add an entry to input file
        in_file_obj.write("{} {} {}\n".format(local_video_file, frame_num, category_zero_or_one))

        # - add an entry to output file
        out_file_obj.write("{0}/{1:07d}\n".format(outdir_plus_video_id, frame_num))


    # - TODO(chuck): generate a bash script that runs c3d feature extraction

    if not os.path.isdir(outdir_plus_video_id):
        os.makedirs(outdir_plus_video_id)

in_file_obj.close()
out_file_obj.close()
