#!/usr/bin/env python

'''
Using fc6 activation features from C3D network (pretrained to UCF101 activity
recognition task)
'''

import numpy as np
import array
import sys
import os
import pylab
#from tsne import bh_sne
from sklearn.manifold import TSNE as tsne

c3d_feature_dir = r'/media/TB/Videos/youtube-objects-all/c3d_features'

def extract_and_save_fc6_features():
    # allow overwrite
    #force_overwite = True
    force_overwite = False

    for dirname in os.listdir(c3d_feature_dir):
        shot_fullpath = os.path.join(c3d_feature_dir, dirname)
        if os.path.isdir(shot_fullpath):
            print "-"*79
            print "[Info] Processing shot={}...".format(dirname)

            num_frame = 0
            for c3d_feature in os.listdir(shot_fullpath):
                if c3d_feature.endswith('.fc6-1'):
                    #print "frame={}".format(c3d_feature)
                    c3d_feature_fullpath = os.path.join(shot_fullpath, c3d_feature)
                    f = open(c3d_feature_fullpath, "rb") # read binary data
                    s = f.read() # read all bytes into a string
                    f.close()
                    (n, c, l, h, w) = array.array("i", s[:20])
                    fc6 = np.array(array.array("f", s[20:]))
                    #print "n={}, c={}, l={}, h={}, w={}".format(n, c, l, h, w)
                    #print "fc6[:5]={}".format(fc6[:5])
                    if num_frame == 0:
                        fc6_avg = fc6
                    else:
                        fc6_avg += fc6
                    num_frame += 1

            fc6_avg /= num_frame
            #print "fc6_avg[:5]={}, num_frame={}".format(fc6_avg[:5], num_frame)

            shot_feature_filename = shot_fullpath + '.csv'
            if os.path.isfile(shot_feature_filename) and not force_overwite:
                print "[Warning] feature was already saved. Skipping this shot..."
                continue
            else:
                print "[Info] saving fc6 feature as {}".format(shot_feature_filename)
                tmp = fc6_avg.reshape(1, fc6_avg.shape[0])
                #np.savetxt(shot_feature_filename, fc6_avg);
                np.savetxt(shot_feature_filename, tmp, fmt='%.8f', delimiter='\n')


def save_all_features():
    ''' '''

    all_features_file = os.path.join(c3d_feature_dir, 'all_features.npy')
    all_labels_file   = os.path.join(c3d_feature_dir, 'all_labels.npy')

    if os.path.isfile(all_features_file) and os.path.isfile(all_labels_file):
        X = np.load(all_features_file)
        all_labels = np.load(all_labels_file)

    else:

        num_shots = 0
        all_labels = []
        for filename in os.listdir(c3d_feature_dir):

            if not filename.endswith('.csv'):
                continue

            c3d_feature_fullpath = os.path.join(c3d_feature_dir, filename)
            print "[Info] Processing shot={}...".format(c3d_feature_fullpath)
            c3d_feature = np.loadtxt(c3d_feature_fullpath)

            if num_shots == 0:
                X = c3d_feature
            else:
                X = np.vstack((X, c3d_feature))
            label = filename.split('_')[0]
            all_labels.append(label)
            num_shots += 1

            #print "X.shape={}, len(all_labels)={}, num_shots={}".format(X.shape, len(all_labels), num_shots)

        np.save(all_features_file, X)
        np.save(all_labels_file, all_labels)

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

    #tsne_output_file = os.path.join(c3d_feature_dir, 'tsne_output.npy')
    tsne_output_file = os.path.join(c3d_feature_dir, 'Y_bhtsne.txt')
    if os.path.isfile(tsne_output_file):
        #Y = np.load(tsne_output_file)
        Y = np.loadtxt(tsne_output_file)
    else:
        tsne_model = tsne(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne_model.fit_transform(X)
        np.save(tsne_output_file, Y)

    #pylab.scatter(Y[:,0], Y[:,1], 20, all_labels_indexed);
    #pylab.savefig('tsne_video_clusters.png')
    #pylab.show()

    for count, label in enumerate(uniq_labels):
        this_label_ind = np.where(all_labels == label)[0]
        #pylab.scatter(Y[this_label_ind,0], Y[this_label_ind,1], 20, count)
        print "count={}, label={}, len(this_label_ind)={}".format(count, label, len(this_label_ind))
        if count < 6:
            marker = 'o'
        else:
            marker = 'v'
        pylab.plot(Y[this_label_ind,0], Y[this_label_ind,1], marker=marker, linestyle='', ms=6, label=label);
    pylab.legend(numpoints=1, loc='upper left')
    #pylab.xlim([-23, 15])
    pylab.xlim([-80, 60])
    pylab.savefig('tsne_video_clusters.png')
    pylab.show()

def main():

    #extract_and_save_fc6_features()
    X, all_labels = save_all_features()
    visualize_clusters(X, all_labels)

if __name__ == "__main__":
    main()
