#include <opencv2/opencv.hpp>
#include "src/yolov5_cls.h"

int main()
{
    std::vector<float> result;
    std::vector<std::string> labels = {"n0","n1","n2","n3","n4","n5","n6","n7","n8","n9"};
    std::string onnx_file = "/media/yao/Data/yolov5_cls_trt/models/best.onnx";
    cv::Mat org_img = cv::imread("/media/yao/Data/yolov5_cls_trt/images/n101.jpg");
    YOLOV5_cls YOLOV5_cls(onnx_file);
    YOLOV5_cls.Init_Model();
    result = YOLOV5_cls.Inferimg(org_img);
    std::vector<float>::iterator max_ele = std::max_element(std::begin(result), std::end(result));
    int max_pos = std::distance(std::begin(result), max_ele);
    std::cout << "max confidence is " << *max_ele<< " ,label is " << labels[max_pos] << std::endl;

    return 0;
}
