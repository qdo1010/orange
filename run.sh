DIR_FFMPEG=$HOME/nvidia/ffmpeg
DIR_TENSORRT=$HOME/nvidia/TensorRT
# CUDA_MODULE_LOADING=LAZY: load GPU kernels on first use — less GPU memory and
# faster TensorRT init (silences the "lazy loading is not enabled" warning).
sudo CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/usr/local/cuda/lib64:$DIR_FFMPEG/build/lib:$DIR_TENSORRT/lib ./targets/orange