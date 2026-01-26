

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

将`/opt/tpuv7/driver-1.7.0/tpuv7/sc11_config.ini`和`/opt/tpuv7/driver-1.7.0/tpuv7/sc11_config_chip2.ini`中scheduler的值修改为`reuse`。

## 3. Run ByteMLPerf for TPU backend

```bash
python3  launch.py --task widedeep-onnx-fp32 --hardware_type TPU
python3  launch.py --task resnet50-torch-fp32 --hardware_type TPU
```

# Models supported 

| Model name                  |  Precision | QPS       | Dataset          | Metric name | Metric value | report |
| ----                        | ----       | ----      | ----             | ----        | ----     | ---- |
| widedeep-tf-fp32            | FP16       | 1301171 | Open Criteo Kaggle | Top-1       | 0.77392  | [report](../../reports/TPU/widedeep-tf-fp32/) |
| resnet50-torch-fp32         | INT8       | 25189   | Open Imagenet      | Top-1       | 0.7686   | [report](../../reports/TPU/resnet50-torch-fp32/) |
| videobert-onnx-fp32         | FP32       | 349     | Open Cifar         | Top-1       | 0.6171   | [report](../../reports/TPU/videobert-onnx-fp32/) |
| conformer-encoder-onnx-fp32 | FP16       | 1940    | None               | -           | -        | [report](../../reports/TPU/conformer-encoder-onnx-fp32/) |
| yolov5-onnx-fp32            | FP16       |  1035   | None               | -           | -        | [report](../../reports/TPU/yolov5-onnx-fp32/) |
| bert-torch-fp32             | FP32       |  43     | open_squad         | F1-score    | 91.2037  | [report](../../reports/TPU/bert-torch-fp32/) |
| albert-torch-fp32           | FP32       |  65     | open_squad         | F1-score    | 87.80423 | [report](../../reports/TPU/albert-torch-fp32/) |
| roberta-torch-fp32          | FP32       |  66     | open_squad         | F1-score    | 94.68039 | [report](../../reports/TPU/roberta-torch-fp32/) |
| roformer-tf-fp32            | FP32       |  84     | open_cali2019      | Top-1       | 0.64974  | [report](../../reports/TPU/roformer-torch-fp32/) |
| deberta-torch-fp32          | FP16       |   2     | open_squad         | F1-score    | 90.89661 | [report](../../reports/TPU/deberta-torch-fp32/) |
| swin-large-torch-fp32       | FP16       |  122    | open_imagenet      | Top-1       | 0.846    | [report](../../reports/TPU/swin-large-torch-fp32/) |
| unet-onnx-fp32              | FP32       |  39     | None               |   -         |  -       | [report](../../reports/TPU/unet-onnx-fp32/) |
| vae-decoder-onnx-fp32       | FP32       |  15     | None               | -           | -        | [report](../../reports/TPU/vae-decoder-onnx-fp32/) |
| vae-encoder-onnx-fp32       | FP32       |  34     | None               | -           | -        | [report](../../reports/TPU/vae-decoder-onnx-fp32/) |
| clip-onnx-fp32              | FP32       |  506    | None               | -           | -        | [report](../../reports/TPU/clip-onnx-fp32/) |