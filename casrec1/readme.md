## Preparation
1) The code is tested on an NVIDIA Tesla T4 Platform.

2) The code is based on pytorch 1.10.0 and pytorch-lightning 1.5.3 in ubuntu server. Required python packages:

	* python==3.9
	* torch==1.10.0
	* pytorch-lightning==1.5.3
	* numpy==1.25.2
    * tensorboard==2.8.0

3) The dataset here is a subset of Taobao dataset inside the "supplementary/casr/data" folder, which contains only users with 10-50 interactions .

## Quick Start

python main.py