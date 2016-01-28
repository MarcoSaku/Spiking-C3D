#!/usr/bin/env python

'''
Using fc6 activation features from C3D network (pretrained to UCF101 activity
recognition task)
'''

import csv
import glob
import numpy as np
import array
import sys
import os
import pylab
#from tsne import bh_sne
from sklearn.manifold import TSNE as tsne

c3d_feature_dir = r'/home/chuck/c3d_features2'
video_list_file = 'dog_cat_streams.csv'

def extract_and_save_fc6_features():
    # allow overwrite
    #force_overwite = True
    force_overwite = False

    videos = []
    f = open(video_list_file, 'rt')
    try:
        reader = csv.reader(f)
        for row in reader:
            video_url = row[0].split('/')[-1]
            category = int(row[1])
            #print "[Info] video_url='{}', category={}...".format(video_url, category)
            videos.append((video_url, category))
    finally:
        f.close()

    for dirname in os.listdir(c3d_feature_dir):
        video_fullpath = os.path.join(c3d_feature_dir, dirname)
        if os.path.isdir(video_fullpath):
            print "-"*79
            print "[Info] Processing video={}...".format(dirname)

            all_files = list(glob.iglob(os.path.join(video_fullpath, '*.fc6-1')))
            if len(all_files) == 0:
                continue

            try:
                all_frames = [int(os.path.basename(x).replace(".fc6-1","")) for x in all_files]
            except ValueError:
                print "[Error] Somehow fc6-1 files contain invalid characters"
                continue

            # sanity check
            last_frame = max(all_frames)

            #if len(all_frames) != last_frame/100 + 1:
            #    print "[Error] Number of frames look strange..."
            #    continue
            #else:
            #    (min_frame, max_frame) = np.percentile(all_frames, [40.0, 60.0])

            min_frame = 0
            max_frame = 500

            #print "all_frames={}".format(all_frames)
            #print "min_frame={}, max_frame={}".format(min_frame, max_frame)

            # check if this video is tagged with dog or(/and) cat
            tag = [x[1] for x in videos if x[0] == dirname]
            #print "tag={}".format(tag)
            if len(tag) == 1 and tag[0] == 9:
                category = "dog"
            elif len(tag) == 1 and tag[0] == 21:
                category = "cat"
            elif len(tag) == 2:
                category = "both"
            else:
                category = "unknown"

            num_frame = 0
            for c3d_feature in os.listdir(video_fullpath):
                if c3d_feature.endswith('.fc6-1'):
                    #print "frame={}".format(c3d_feature)
                    frame_num = int(c3d_feature.replace(".fc6-1",""))

                    if frame_num < min_frame or frame_num > max_frame:
                        continue

                    c3d_feature_fullpath = os.path.join(video_fullpath, c3d_feature)
                    f = open(c3d_feature_fullpath, "rb") # read binary data
                    s = f.read() # read all bytes into a string
                    f.close()
                    (n, c, l, h, w) = array.array("i", s[:20])
                    fc6 = np.array(array.array("f", s[20:]))
                    #print "n={}, c={}, l={}, h={}, w={}".format(n, c, l, h, w)
                    #print "fc6[:5]={}".format(fc6[:5])

                    # some elements in fc6 can contain "inf" or "nan" (but why?)
                    if np.any(np.isnan(fc6)) or np.any(np.isinf(fc6)):
                        continue

                    if num_frame == 0:
                        fc6_avg = fc6
                    else:
                        fc6_avg += fc6
                    num_frame += 1

            if num_frame == 0:
                continue

            fc6_avg /= num_frame
            #print "fc6_avg[:5]={}, num_frame={}".format(fc6_avg[:5], num_frame)

            video_feature_filename = os.path.join(c3d_feature_dir, category + "_" + dirname + '.csv')
            if os.path.isfile(video_feature_filename) and not force_overwite:
                print "[Warning] feature was already saved. Skipping this video..."
                continue
            else:
                print "[Info] saving fc6 feature as {}".format(video_feature_filename)
                tmp = fc6_avg.reshape(1, fc6_avg.shape[0])
                np.savetxt(video_feature_filename, tmp, fmt='%.8f', delimiter='\n')

def save_all_features():
    ''' '''

    all_features_file = os.path.join(c3d_feature_dir, 'all_features.npy')
    all_labels_file   = os.path.join(c3d_feature_dir, 'all_labels.npy')
    all_videos_file   = os.path.join(c3d_feature_dir, 'all_videos.txt')

    if os.path.isfile(all_features_file) and os.path.isfile(all_labels_file):
        X = np.load(all_features_file)
        all_labels = np.load(all_labels_file)

    else:
        num_videos = 0
        all_labels = []
        all_videos = []
        for filename in os.listdir(c3d_feature_dir):

            if not filename.endswith('.csv'):
                continue

            c3d_feature_fullpath = os.path.join(c3d_feature_dir, filename)
            print "[Info] Processing video={}...".format(c3d_feature_fullpath)
            c3d_feature = np.loadtxt(c3d_feature_fullpath)

            if num_videos == 0:
                X = c3d_feature
            else:
                X = np.vstack((X, c3d_feature))
            label = filename.split('_')[0]
            all_labels.append(label)
            all_videos.append(filename.replace(".csv",""))
            num_videos += 1

            #print "X.shape={}, len(all_labels)={}, num_videos={}".format(X.shape, len(all_labels), num_videos)

        np.save(all_features_file, X)
        np.save(all_labels_file, all_labels)
        np.savetxt(all_videos_file, all_videos, '%s')

    return X, all_labels

#    print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
#    print "Running example on 2,500 MNIST digits..."
#    X = Math.loadtxt("mnist2500_X.txt");
#    labels = Math.loadtxt("mnist2500_labels.txt");
#    Y = tsne(X, 2, 50, 20.0);
#    Plot.scatter(Y[:,0], Y[:,1], 20, labels);
#    Plot.show()

def visualize_clusters(X, all_labels):
    ''' '''

    # sort of like "unique"
    uniq_labels = list(set(all_labels))
    all_labels_indexed = [uniq_labels.index(x) for x in all_labels]

    tsne_output_file = os.path.join(c3d_feature_dir, 'Y_tsne_dog_cat.npy')
    tsne_output_txt_file = os.path.join(c3d_feature_dir, 'Y_tsne_dog_cat.txt')
    if os.path.isfile(tsne_output_file):
        #Y = np.load(tsne_output_file)
        Y = np.load(tsne_output_file)
    else:
        tsne_model = tsne(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne_model.fit_transform(X)
        np.save(tsne_output_file, Y)
        np.savetxt(tsne_output_txt_file, Y)

    #pylab.scatter(Y[:,0], Y[:,1], 20, all_labels_indexed);
    #pylab.savefig('tsne_video_clusters.png')
    #pylab.show()

    for count, label in enumerate(uniq_labels):
        # skip "both" label
        if label == 'both':
            continue
        this_label_ind = np.where(all_labels == label)[0]
        #pylab.scatter(Y[this_label_ind,0], Y[this_label_ind,1], 20, count)
        print "label={}, len(this_label_ind)={}".format(label, len(this_label_ind))
        if count < 6:
            marker = 'o'
        else:
            marker = 'v'

        if count == 1:
            color='g'
        elif count == 2:
            color = 'r'

        pylab.plot(Y[this_label_ind,0], Y[this_label_ind,1], marker=marker, linestyle='', ms=6, label=label, color=color);
    pylab.legend(numpoints=1, loc='upper left')
    pylab.xlim([-23, 17])
    pylab.ylim([-16, 16])
    pylab.grid()
    #pylab.xlim([-80, 60])
    pylab.savefig('tsne_cat_dog_clusters.png')
    pylab.show()

def main():

    #extract_and_save_fc6_features()
    X, all_labels = save_all_features()
    visualize_clusters(X, all_labels)

if __name__ == "__main__":
    main()
