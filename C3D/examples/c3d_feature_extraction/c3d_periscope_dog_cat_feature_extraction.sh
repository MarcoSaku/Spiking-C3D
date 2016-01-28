echo pwd=`pwd`

GLOG_logtosterr=1 \
  ../../build/tools/extract_image_features.bin \
  prototxt/c3d_periscope_dog_cat_feature_extractor.prototxt \
  conv3d_deepnetA_sport1m_iter_1900000 \
  0 40 1000000 \
  prototxt/periscope_dog_cat_output_list_video_prefix_scene_segmented.txt \
  fc6-1

#  fc7-1 fc6-1 prob
