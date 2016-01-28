// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/proto/caffe.pb.h"

namespace c3d_caffe {

template <typename Dtype>
class Blob {
 public:
  Blob()
       : num_(0), channels_(0), length_(0), height_(0), width_(0), count_(0), data_(),
       diff_() {}
  explicit Blob(const int num, const int channels, const int length, const int height,
    const int width);

  explicit Blob(const int num, const int channels, const int height,
    const int width);

  void Reshape(const int num, const int channels, const int length, const int height,
    const int width);

  // for backward compatibility
  void Reshape(const int num, const int channels, const int height,
    const int width);

  void ReshapeLike(const Blob& other);
  inline int num() const { return num_; }
  inline int channels() const { return channels_; }
  inline int length() const { return length_; }
  inline int height() const { return height_; }
  inline int width() const { return width_; }
  inline int count() const {return count_; }

  // for backward compatibility
  inline int offset(const int n){
	  return offset(n, 0, 0, 0, 0);
  }
  // for backward compatibility
  inline int offset(const int n, const int c){
	  return offset(n, c, 0, 0, 0);
  }

  inline int offset(const int n, const int c, const int l, const int h,
      const int w) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num_);
    CHECK_GE(c, 0);
    CHECK_LE(c, channels_);
    CHECK_GE(l, 0);
    CHECK_LE(l, length_);
    CHECK_GE(h, 0);
    CHECK_LE(h, height_);
    CHECK_GE(w, 0);
    CHECK_LE(w, width_);
    return (((n * channels_ + c) * length_ + l) * height_ + h) * width_ + w;
  }

  inline int offset(const int n, const int c, const int h,
      const int w) const {
	  return offset(n, c, 0, h, w);
  }
  // Copy from source. If copy_diff is false, we copy the data; if copy_diff
  // is true, we copy the diff.
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  inline Dtype data_at(const int n, const int c, const int l, const int h,
      const int w) const {
    return *(cpu_data() + offset(n, c, l, h, w));
  }

  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return *(cpu_data() + offset(n, c, 0, h, w));
  }

  inline Dtype diff_at(const int n, const int c, const int l, const int h,
      const int w) const {
    return *(cpu_diff() + offset(n, c, l, h, w));
  }

  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return *(cpu_diff() + offset(n, c, 0, h, w));
  }

  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);
  const Dtype* gpu_data() const;
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  void Update();
  void FromProto(const BlobProto& proto);
  void Lift3DFromProto(const BlobProto& proto, const int l);
  void ToProto(BlobProto* proto, bool write_diff = false) const;

  // Set the data_/diff_ shared_ptr to point to the SyncedMemory holding the
  // data_/diff_ of Blob other -- useful in layers which simply perform a copy
  // in their forward or backward pass.
  // This deallocates the SyncedMemory holding this blob's data/diff, as
  // shared_ptr calls its destructor when reset with the = operator.
  void ShareData(const Blob& other);
  void ShareDiff(const Blob& other);

 protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  int num_;
  int channels_;
  int length_;
  int height_;
  int width_;
  int count_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace c3d_caffe

#endif  // CAFFE_BLOB_HPP_
