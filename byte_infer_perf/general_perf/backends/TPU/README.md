

# How to run

## 1. Create docker container

```bash
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name TPUPerf -td -v /dev/:/dev/ -v /opt/:/opt/ -v <your path>:/workspace/ --entrypoint bash sophgo/tpuc_dev:latest
docker exec -it TPUPerf bash
```

## 2. Environment Initialization

```bash
pip3 install tpu_mlir
apt install unzip
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Stable_diffusion_3/BM1690/sophon-sail2.zip
unzip sophon-sail2.zip
# 依照sail2目录下的README，在当前环境编译出whl并安装
```

## 3. Run ByteMLPerf for TPU backend

```bash
python3  launch.py --task widedeep-onnx-fp32 --hardware_type TPU
python3  launch.py --task resnet50-torch-fp32 --hardware_type TPU
```

# Models supported 

| Model name          |  Precision | QPS       | Dataset            | Metric name | Metric value | report |
| ----                | ----       | ----      | ----               | ----        | ----     | ---- |
| widedeep-tf-fp32    | FP16       | 118478    | Open Criteo Kaggle | Top-1       | 0.77392 | [report](../../reports/TPU/widedeep-tf-fp32/) |
| resnet50-torch-fp32 | INT8       | 2049      | Open Imagenet      | Top-1       | 0.76963 | [report](../../reports/IPU/resnet50-torch-fp32/) |
| videobert-onnx-fp32 | FP32       | 44        | Open Cifar      | Top-1       | 0.6171 | [report](../../reports/IPU/videobert-onnx-fp32/) |
| conformer-encoder-onnx-fp32 | FP16 | 571     | None             | -           | - | [report](../../reports/IPU/conformer-encoder-onnx-fp32/) |
| yolov5-onnx-fp32    | FP16        |  231     | None             | -           | - | [report](../../reports/IPU/yolov5-onnx-fp32/) |