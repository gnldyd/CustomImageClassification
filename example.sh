#!/bin/bash

#python3 main.py --gpu_parallel=True --data_path=./data/fruit/ --batch_size=5 --model_name=alexnet --classes=5 --epochs=10
#python3 main.py --gpu_parallel=True --data_path=./data/fruit/ --batch_size=5 --model_name=vgg --model_pretrained=False --classes=5 --epochs=3 --predict=True
#python3 main.py --gpu_index=0 --data_path=./data/fruit/ --batch_size=5 --model_name=resnet --model_pretrained=False --classes=5 --epochs=5 --predict=True
python3 main.py --gpu_index=0 --data_path=./data/fruit/ --batch_size=5 --model_name=MNasNet1_0 --model_pretrained=False --classes=5 --epochs=5 --predict=False
