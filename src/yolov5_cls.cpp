#include "yolov5_cls.h"
#include <chrono>
#include <algorithm>

YOLOV5_cls::YOLOV5_cls(std::string onnx_model){
    onnx_file = onnx_model;
    engine_file = onnx_model.replace(onnx_model.find("onnx"),4,"engine");
    BATCH_SIZE = 1;
    INPUT_CHANNEL = 3;
    IMAGE_WIDTH = 224;
    IMAGE_HEIGHT = 224;
    OUTPUT_SIZE = 10;
}

YOLOV5_cls::~YOLOV5_cls()
{
    Destory_Model();
}

void YOLOV5_cls::onnx2engine(std::string onnxfilePath, std::string enginefilePath,ICudaEngine*& engine_model, int type) {
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    auto parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(onnxfilePath.c_str(), 2);
    for (int i = 0; i < parser->getNbErrors(); ++i)
    {
          std::cout << "load error: "<< parser->getError(i)->desc() << std::endl;
    }
    printf("tensorRT load onnx model successfully!\n");

    // 创建推理引擎
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(16 * (1 << 20));
    if (type == 1) {
          config->setFlag(BuilderFlag::kFP16);
    }
    if (type == 2) {
          config->setFlag(BuilderFlag::kINT8);
    }
    engine_model = builder->buildEngineWithConfig(*network, *config);
    assert(engine_model != nullptr);
    context = engine_model->createExecutionContext();
    assert(context != nullptr);
    std::cout << "try to save engine file" << std::endl;
    std::ofstream p(enginefilePath, std::ios::binary);
    if (!p) {
          std::cerr << "could not open plan output file" << std::endl;
          return;
    }
    IHostMemory* modelStream = engine_model->serialize();
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    p.close();
    //modelStream->destroy();
    //myengine->destroy();
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
    std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
}

void YOLOV5_cls::readenginefile(std::string engineFile, ICudaEngine*& engine_model) {
    std::fstream file;
    std::cout << "loading filename from:" << engineFile << std::endl;
    file.open(engineFile, std::ios::binary | std::ios::in);
    file.seekg(0, std::ios::end);                     // 定位到 fileObject 的末尾
    int length = file.tellg();
    file.seekg(0, std::ios::beg);                // 定位到 fileObject 的开头
    std::unique_ptr<char[]> data(new char[length]);
    file.read(data.get(), length);
    file.close();

    IRuntime* trtRuntime = createInferRuntime(gLogger.getTRTLogger());
    assert(trtRuntime != nullptr);
    engine_model = trtRuntime->deserializeCudaEngine(data.get(), length, nullptr);
    assert(engine_model != nullptr);
    context = engine_model->createExecutionContext();
    assert(context != nullptr);
    std::cout << "Read Success" << std::endl;
    //trtModelStream = engine->serialize();
    trtRuntime->destroy();
}

void YOLOV5_cls::Init_Model() {
    // create and load engine
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine) {
        readenginefile(engine_file, engine);
        assert(engine != nullptr);
    } else {
        onnx2engine(onnx_file, engine_file, engine, 1);
        assert(engine != nullptr);
    }
}

void YOLOV5_cls::Destory_Model()
{
    assert(engine != nullptr);
    assert(context != nullptr);

    // destroy the engine
    context->destroy();
    engine->destroy();
}

void YOLOV5_cls::preprocess(cv::Mat &imgs, float* input_data) {
    cv::Mat img_ori;
    cv::resize(imgs, img_ori, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
    std::vector<cv::Mat> InputImage;
    InputImage.push_back(img_ori);
    int ImgCount = InputImage.size();
    for (int b = 0; b < ImgCount; b++) {
        cv::Mat img = InputImage.at(b);
        int w = img.cols;
        int h = img.rows;
        int i = 0;
        for (int row = 0; row < h; ++row) {
            uchar* uc_pixel = img.data + row * img.step;
            for (int col = 0; col < img.cols; ++col) {
                input_data[b * 3 * img.rows * img.cols  + i] = ((float)uc_pixel[2] / 255.0 - 0.485) / 0.229;
                input_data[b * 3 * img.rows * img.cols + i + img.rows * img.cols] = ((float)uc_pixel[1] / 255.0 - 0.456) / 0.224;
                input_data[b * 3 * img.rows * img.cols + i + 2 * img.rows * img.cols] = ((float)uc_pixel[0] / 255.0 - 0.406) / 0.225;
                uc_pixel += 3;
                ++i;
              }
        }

    }
}


void YOLOV5_cls::doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& myengine = context.getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(myengine.getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = myengine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = myengine.getBindingIndex(OUTPUT_BLOB_NAME);
    //const int inputIndex = 0;
    //const int outputIndex = 1;
//     Create GPU buffers on device
    cudaMalloc(&buffers[inputIndex], batchSize * 3 * IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float));
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

std::vector<float> YOLOV5_cls::softmax(float *prob, int n) {
    std::vector<float> res;
    float sum = 0.0f;
    float t;
    for (int i = 0; i < n; i++) {
    t = expf(prob[i]);
    res.push_back(t);
    sum += t;
    }
    for (int i = 0; i < n; i++) {
    res[i] /= sum;
    }
    return res;
}

std::vector<float> YOLOV5_cls::Inferimg(cv::Mat &src_img) {

    cv::Mat ori_img = src_img.clone();
    float img_data[BATCH_SIZE * INPUT_CHANNEL* IMAGE_WIDTH * IMAGE_HEIGHT];
    float prob_result[OUTPUT_SIZE];
    std::vector<float> boxes;
    // preprocess image
    auto pre_start = std::chrono::high_resolution_clock::now();
    preprocess(ori_img, img_data);
    auto pre_end = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(pre_end - pre_start).count();
    std::cout << "preprocess take: " << total_pre << " ms." << std::endl;

    // do inference
    auto t_start = std::chrono::high_resolution_clock::now();
    doInference(*context, img_data, prob_result, BATCH_SIZE);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Inference take: " << total_inf << " ms." << std::endl;

    boxes = softmax(prob_result,OUTPUT_SIZE);

    return boxes;
}
