#ifndef YOLOV5_CLS_TRT_H
#define YOLOV5_CLS_TRT_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <dirent.h>
#include "NvOnnxParser.h"
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include "logging.h"
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include "cuda_runtime_api.h"
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;
static Logger gLogger;


class YOLOV5_cls
{

public:
    YOLOV5_cls(std::string onnx_model);
    ~YOLOV5_cls();
    void Init_Model();
    std::vector<float> Inferimg(cv::Mat &src_img);
    void Destory_Model();

private:

    std::vector<float> softmax(float *prob, int n);
    void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
    void preprocess(cv::Mat &imgs, float* input_data);
    void onnx2engine(std::string onnxfilePath, std::string enginefilePath, ICudaEngine*& engine_model, int type);
    void readenginefile(std::string engineFile, ICudaEngine*& engine_model);
    std::string onnx_file;
    std::string engine_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    int OUTPUT_SIZE;
    const char* INPUT_BLOB_NAME = "images";
    const char* OUTPUT_BLOB_NAME = "output0";
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
};

#endif //YOLOV5_CLS_TRT_H
