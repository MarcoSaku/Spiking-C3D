mkdir -p /media/TB/Videos/youtube-objects-all/c3d_features

#GLOG_logtosterr=1 \
#  ../../build/tools/extract_image_features.bin \
#  prototxt/c3d_sport1m_feature_extractor_video.prototxt \
#  conv3d_deepnetA_sport1m_iter_1900000 \
#  0 50 1 \
#  prototxt/output_list_video_prefix.txt \
#  fc7-1 fc6-1 prob

GLOG_logtosterr=1 \
  ../../build/tools/extract_image_features.bin \
  prototxt/c3d_youtube_objects_feature_extractor.prototxt \
  conv3d_deepnetA_sport1m_iter_1900000 \
  0 40 1000000 \
  prototxt/youtube_objects_output_list_video_prefix.txt \
  fc7-1 fc6-1 prob
