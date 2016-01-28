// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_RNG_CPP_HPP_
#define CAFFE_RNG_CPP_HPP_

#include <boost/random/mersenne_twister.hpp>
#include "caffe/common.hpp"

namespace c3d_caffe {

  typedef boost::mt19937 rng_t;

  inline rng_t* caffe_rng() {
    return static_cast<c3d_caffe::rng_t*>(Caffe::rng_stream().generator());
  }

}  // namespace c3d_caffe

#endif  // CAFFE_RNG_HPP_
