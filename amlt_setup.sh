#!/bin/bash
python --version
pwd
ls
# install from requirements.txt
pip install --user kipoi==0.8.6
pip install --user kipoiseq==0.7.1
pip install --user -r requirements.txt

pip install --user opencv-python-headless numpy decord tqdm omegaconf pytz scikit-learn
# pip install --user absl-py==2.1.0 aiohappyeyeballs==2.4.4 aiohttp==3.11.9 aiosignal==1.3.1 antlr4-python3-runtime==4.9.3 appdirs==1.4.4 async-timeout==5.0.1 attrs==24.2.0 av==12.0.0 beartype==0.18.2 certifi==2024.8.30 charset-normalizer==3.4.0 click==8.1.7 contourpy==1.3.1 cycler==0.12.1 decord==0.6.0 docker-pycreds==0.4.0 einops==0.8.0 filelock==3.16.1 fonttools==4.55.1 frozenlist==1.5.0 fsspec==2024.10.0 gitdb==4.0.11 gitpython==3.1.43 grpcio==1.68.1 idna==3.10 imageio==2.34.0 jinja2==3.1.4 kiwisolver==1.4.7 lightning==2.2.4 lightning-utilities==0.11.9 markdown==3.7 markdown-it-py==3.0.0 markupsafe==3.0.2 matplotlib==3.8.4 mdurl==0.1.2 mpmath==1.3.0 multidict==6.1.0 natsort==8.4.0 networkx==3.4.2 numpy==1.26.4 nvidia-cublas-cu12==12.1.3.1 nvidia-cuda-cupti-cu12==12.1.105 nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-runtime-cu12==12.1.105 nvidia-cudnn-cu12==8.9.2.26 nvidia-cufft-cu12==11.0.2.54 nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 nvidia-cusparse-cu12==12.1.0.106 nvidia-nccl-cu12==2.19.3 nvidia-nvjitlink-cu12==12.6.85 nvidia-nvtx-cu12==12.1.105 omegaconf==2.3.0 opencv-python==4.6.0.66 packaging==24.2 pandas==2.1.4 pillow==11.0.0 propcache==0.2.1 protobuf==4.25.5 psutil==6.1.0 pygments==2.18.0 pyparsing==3.2.0 python-dateutil==2.9.0.post0 pytorch-lightning==2.2.4 pytz==2024.2 pyyaml==6.0.2 requests==2.32.3 rich==13.5.3 safetensors==0.4.2 sentry-sdk==2.19.0 setproctitle==1.3.4 six==1.17.0 smmap==5.0.1 sympy==1.13.3 tensorboard==2.16.2 tensorboard-data-server==0.7.2 torch==2.2.2 torchmetrics==1.6.0 torchvision==0.17.2 tqdm==4.67.1 triton==2.2.0 typing-extensions==4.12.2 tzdata==2024.2 urllib3==2.2.3 wandb==0.16.6 werkzeug==3.1.3 yarl==1.18.3
pip install --user trl==0.13.0 transformers==4.50.3 accelerate==1.6.0 datasets==3.5.0 wandb

pip install ai2-olmo
####### For dnabert 2 #######
# pip install --user transformers==4.29.0
pip install tiktoken gdown tiktoken datasets wandb
pip uninstall -y triton
pip uninstall -y triton --user
#############################
# pip install --user -r HunyuanVideo-I2V/requirements.txt
# pip install flash-attn --no-build-isolation
# conda env create -f VidTok/environment.yaml
# pip install --user trl transformers accelerate datasets
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

which accelerate
