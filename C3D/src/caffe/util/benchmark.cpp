// Copyright 2014 BVLC and contributors.

#include <boost/date_time/posix_time/posix_time.hpp>
#include <cuda_runtime.h>

#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"

namespace c3d_caffe {

Timer::Timer()
    : initted_(false),
      running_(false),
      has_run_at_least_once_(false) {
  Init();
}

Timer::~Timer() {
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaEventDestroy(start_gpu_));
    CUDA_CHECK(cudaEventDestroy(stop_gpu_));
  }
}

void Timer::Start() {
  if (!running()) {
    if (Caffe::mode() == Caffe::GPU) {
      CUDA_CHECK(cudaEventRecord(start_gpu_, 0));
    } else {
      start_cpu_ = boost::posix_time::microsec_clock::local_time();
    }
    running_ = true;
    has_run_at_least_once_ = true;
  }
}

void Timer::Stop() {
  if (running()) {
    if (Caffe::mode() == Caffe::GPU) {
      CUDA_CHECK(cudaEventRecord(stop_gpu_, 0));
      CUDA_CHECK(cudaEventSynchronize(stop_gpu_));
    } else {
      stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    }
    running_ = false;
  }
}

float Timer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds_, start_gpu_,
                                    stop_gpu_));
  } else {
    elapsed_milliseconds_ = (stop_cpu_ - start_cpu_).total_milliseconds();
  }
  return elapsed_milliseconds_;
}

float Timer::Seconds() {
  return MilliSeconds() / 1000.;
}

void Timer::Init() {
  if (!initted()) {
    if (Caffe::mode() == Caffe::GPU) {
      CUDA_CHECK(cudaEventCreate(&start_gpu_));
      CUDA_CHECK(cudaEventCreate(&stop_gpu_));
    }
    initted_ = true;
  }
}

}  // namespace c3d_caffe
