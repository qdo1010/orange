# Standalone build for the JARVIS pose offline smoke test (no Emergent SDK).
# This is a dev/validation target; the real integration links these .o files
# into lime's CMake build. See docs/jarvis_integration.md.
#
#   make            # build jarvis_pose_offline
#   make run        # build + run against the sample model + calibration

TRT_DIR  ?= /home/ratan/nvidia/TensorRT-8.6.1.6
CUDA_DIR ?= /usr/local/cuda
EIGEN    ?= /usr/include/eigen3

NVCC := $(CUDA_DIR)/bin/nvcc
CXX  := g++

OPENCV_INC ?= /usr/local/include/opencv4
OPENCV_LIB ?= /usr/local/lib
INC := -I$(TRT_DIR)/include -I$(CUDA_DIR)/include -I$(EIGEN) -I$(OPENCV_INC)
CXXFLAGS := -std=c++17 -O2 $(INC)
LDFLAGS  := -L$(TRT_DIR)/lib -L$(CUDA_DIR)/lib64 -L$(OPENCV_LIB) \
            -lnvinfer -lnvinfer_plugin -lcudart \
            -lopencv_core -lopencv_calib3d -lstdc++fs -lpthread

OBJ := jarvis_cuda.o jarvis_calib.o jarvis_pose_offline.o

jarvis_pose_offline: $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

jarvis_cuda.o: jarvis_cuda.cu jarvis_cuda.h
	$(NVCC) -std=c++17 -O2 -arch=sm_86 -I$(CUDA_DIR)/include -c jarvis_cuda.cu -o $@

jarvis_calib.o: jarvis_calib.cpp jarvis_pose.h red_math.h jarvis_trt.h jarvis_cuda.h
	$(CXX) $(CXXFLAGS) -c jarvis_calib.cpp -o $@

jarvis_pose_offline.o: jarvis_pose_offline.cpp jarvis_pose.h red_math.h jarvis_trt.h jarvis_cuda.h
	$(CXX) $(CXXFLAGS) -c jarvis_pose_offline.cpp -o $@

MODEL ?= /home/ratan/src/realtime_jarvis_model/mouse_merge_6kp/onnx
CALIB ?= /home/ratan/src/realtime_jarvis_model/calibration
SERIALS ?= 2002486,2002487,2005325,2006050
run: jarvis_pose_offline
	LD_LIBRARY_PATH=$(TRT_DIR)/lib:$(CUDA_DIR)/lib64:$$LD_LIBRARY_PATH \
	./jarvis_pose_offline $(MODEL) $(CALIB) $(SERIALS) 0,1,2,3 4

clean:
	rm -f $(OBJ) jarvis_pose_offline
