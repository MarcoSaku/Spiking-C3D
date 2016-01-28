GLOG_logtosterr=1 \
  ../../build/tools/extract_image_features.bin \
  ./conv3d_ucf101_feature_extraction.prototxt \
  ./conv3d_ucf101_iter_50000 \
  0 \
  50 \
  41822 \
  ./output.lst \
  prob

# num of clips = 41822
# num of minibatches = 41822 / 40 = 1045
