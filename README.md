# Spiking-C3D
Convolutional 3D Spiking Neural Network to classify videos using Leaky-Integrate-and-Fire Neurons.

The training is done using the 3D Convolutional Neural Network released by Facebook: https://research.facebook.com/blog/c3d-generic-features-for-video-analysis/.

Here we use a Spiking Deep Neural Network to classify the videos: this work is an extension of this publication: http://arxiv.org/abs/1510.08829

The SNN is created using the framework Nengo: http://www.nengo.ca/
## How to use
1. Compile C3D (compile also Python wrapper)
2. Download the video dataset YUPENN (http://vision.eecs.yorku.ca/research/dynamic-scenes/) and extract the frames: the C3D wants the format %06d.jpg for the file image
3. Edit the paths into C3D/examples/c3d_train_yupenn/lst_files/test_yupenn_full_4vid_shuffle.lst
4. Run the script 'run_nengo_3d.py'   
