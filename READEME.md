model 1: 
1x RTX 6000 Ada (48 GB VRAM)
62 GB RAM • 16 vCPU
Total Disk: 40 GB

model 2 : 
1x RTX A6000 (48 GB VRAM)
50 GB RAM • 8 vCPU
Total Disk: 40 GB

create pod in runpod 
open port 8000 

(with required space 30 GB Disk
50 GB Pod Volume if not 80)

git clone url

add model cd repositories/Fooocus/models/loras

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

if successfully istall above then

pip install transformers torch torchvision
pip install av packaging pillow requests
pip install qwen-vl-utils
pip install qwen-vl-utils[decord]
pip install decord
pip install wheel setuptools
pip install flash-attn --no-build-isolation



python main.py --share --always-high-vram --preset anime --port 8000 --host 0.0.0.0

run file for call api
check image and baseurl
python callRequestTesting.py
