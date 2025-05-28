在 https://pytorch.org/get-started/previous-versions/ 上查找下载 torch 版本指令
在 https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html 上寻找 PyTorch 版本、CUDA 版本与 MMCV 版本的对应关系

使用 MIM 安装 MMEngine 和 MMCV。
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet
```

从源码安装 mmdet
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

注意，这里安装的时候一定要找好版本，由于 mmdet 要求 mmcv < 2.2.0，于是我们要在 https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html 上寻找对应的 cuda 和 torch 版本，然后再在 https://pytorch.org/get-started/previous-versions/ 上查找下载指令