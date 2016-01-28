export GOOGLE_LOG_DIR=./log
export GLOG_DIR=./log
export GLOG_logtostderr=1

../../build/tools/finetune_net.bin \
  c3d_dextro_benchmark_solver.prototxt \
  ../../conv3d_deepnetA_sport1m_iter_1900000 \
  2>&1 | tee c3d_dextro_benchmark_log.txt
