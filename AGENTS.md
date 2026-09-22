# 项目说明

您正在为 Seeed reCamera Pro 进行实际的 RKNN 模型基准测试。

## 目标设备

- 设备：Seeed reCamera Pro
- SoC：Rockchip RV1126B
- IP：192.168.3.207
- 用户名：root
- 密码：recamera.1
- 目标架构：aarch64
- 总系统 RAM：2 GB

您正在使用 x86_64 Linux 主机。

您可以使用 SSH、SCP、rsync、git、wget、curl 以及其他合适的工具连接到设备并对其进行操作。

---

## 所需技能

在进行 RV1126B 开发工作之前，请安装并使用：

https://github.com/Seeed-Projects/recamera-pro-development-skill.git

对于 Codex，请使用 `main` 分支。

该技能应安装在：

~/.agents/skills/recamera-rknn-dev

对于 RKNN 转换、交叉编译、部署、运行时调试和基准测试工作，请优先使用以下方式定义的工作流程：

$recamera-rknn-dev

---

## 项目目的

在model/yolo的这个目录下可以找到yolo家族的几个文件夹，你需要进去各个文件夹中，可以看到有onnx模型，你需要对于这个模型进行优化处理，速度尽可能的提升，因为我们是直接从pt导出的onnx模型，和rk的model zoo 中提供的onnx模型不一致的，你现在是需要去比较一下，然后把我们本地的onnx优化，经可能的提升运行的速度。

在提升速度的同时，也需要对于模型的准确度校验，转为rknn之后，把模型推送到pro这个设备上进行推理，推理的图像是bus.jpg这个图像，在现在的本地文件夹也可以找到的，精确度没有明确的要求，只要大致正确就行。最终产出的目录，示例为这样：

就以model/yolo/yolov8_det为例子：

最终的目录下是应该有yolov8n_(模型输入大小)_(模型任务).rknn 、yolov8n.onnx、result_bus.jpg、infer.py、convert_to_rknn.py，以及readme说明，其中对于onnx优化的代码不需要，需要的onnx是一个已经优化好了的onnx模型中见过程不重要。readme中需要说明这个模型的运行速度怎么样，其中的convert_to_rknn.py文件要怎么用，infer.py要怎么用。

其中的infer.py是需要有两个模式，当输入 --input 指定了图像路径的时候，就是使用图像推理，如果没指定的时候就是使用video13这个节点去推理，推理二十次，然后排除最开始的五次推理预热，在最终的运行的结尾输出平均的推理的速度如何。如果是采用摄像头作为输入的话，那就不需要去最帧绘制处理，只需要确保可以推理就行。如果是输入的图像，就需要把帧结果绘制到帧上，因为是指定了图像，绘制的话，只需要在最后一次推理的时候，把结果绘制上就行，并且保存到设备中，然后再拉取下来这个图像保存再本地对应的文件夹中即可。

如果模型不是yolo的模型，请使用这个模型常用的一个图像数据去输入，例如ocr就不适用bus.jpg去推理，此时你需要去网上寻找对应的图像资源来推理
