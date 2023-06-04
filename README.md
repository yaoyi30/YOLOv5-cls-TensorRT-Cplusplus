# YOLOv5-cls-TensorRT-Cplusplus
This is a TensorRT project based on yolov5-cls, using C++
# 1.修改CmakeLists.txt中的cuda和tensorrt路径
![1](https://github.com/yaoyi30/YOLOv5-cls-TensorRT-Cplusplus/assets/56180347/58e9756a-6a5c-431f-bd2a-d9f6b1dd3cbf)
# 2.修改main.py中的labels,onnx_file和ori_img
![2](https://github.com/yaoyi30/YOLOv5-cls-TensorRT-Cplusplus/assets/56180347/0b2599bb-384c-4039-ab80-185bb106cd54)
# 3.修改src/yolov5_cls.cpp中的IMAGE_WIDTH、IMAGE_HEIGHT、OUTPUT_SIZE，分别是输入图像的宽和高还有分类的类别数
![3](https://github.com/yaoyi30/YOLOv5-cls-TensorRT-Cplusplus/assets/56180347/7292af49-777c-49dd-975a-a1cbc3ca70cf)
# 4.将转好的onnx模型放入到models文件夹中
![4](https://github.com/yaoyi30/YOLOv5-cls-TensorRT-Cplusplus/assets/56180347/c0747b66-7f33-4769-88a5-694287c28214)
# 5.开始运行
![2023-06-04 16-17-44屏幕截图](https://github.com/yaoyi30/YOLOv5-cls-TensorRT-Cplusplus/assets/56180347/a7047421-2c47-4466-a389-24852b1a463a)
第一次运行会把onnx模型转为.engine模型，时间稍微长一些
![2023-06-04 16-18-13屏幕截图](https://github.com/yaoyi30/YOLOv5-cls-TensorRT-Cplusplus/assets/56180347/cd24179b-f729-4e22-b6b7-98235e25584d)
之后运行就直接调用.engine模型进行推理
