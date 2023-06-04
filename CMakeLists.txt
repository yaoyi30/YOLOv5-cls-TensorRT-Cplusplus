cmake_minimum_required(VERSION 3.5)

project(yolov5_cls_trt)

# find required opencv
find_package(OpenCV REQUIRED)

# cuda
set(CUDA_ROOT_PATH /usr/local/cuda/) # 根目录
set(CUDA_INCLUDE_DIRS ${CUDA_ROOT_PATH}/include/)   # 头文件夹
set(CUDA_LIB ${CUDA_ROOT_PATH}lib64/libcudart.so)  # 依赖动态库

# tensorrt
set(TENSORRT_ROOT_PATH /home/yao/TensorRT-7.1.3.4/) # 根目录
set(TENSORRT_INCLUDE_DIRS ${TENSORRT_ROOT_PATH}/include/)   # 头文件夹
set(TENSORRT_LIB ${TENSORRT_ROOT_PATH}lib/libnvinfer.so ${TENSORRT_ROOT_PATH}lib/libnvonnxparser.so)  # 依赖动态库

add_executable(yolov5_cls_trt ${PROJECT_SOURCE_DIR}/src/yolov5_cls.cpp ${PROJECT_SOURCE_DIR}/main.cpp)

include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})

target_link_libraries(yolov5_cls_trt ${OpenCV_LIBS} ${TENSORRT_LIB} ${CUDA_LIB})
