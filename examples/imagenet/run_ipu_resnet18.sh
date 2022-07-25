export JAX_PLATFORM_NAME=cpu
python main.py --workdir=./imagenet_resnet18 --config=configs/ipu_x1.py
unset JAX_PLATFORM_NAME