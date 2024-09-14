'''
Description: 
Version: 
Author: zrk
Date: 2023-08-15 17:29:43
LastEditors: zrk
LastEditTime: 2024-07-30 07:53:23
'''
import argparse
import torch
import os
parser = argparse.ArgumentParser()



parser.add_argument('--save_file', default='./save_politics/p5-2_2-shot_correct') # 注意此处定义的p5与之前的区别
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--startpoint', type=int, default=0)
parser.add_argument('--endpoint', type=int, default=232)
parser.add_argument('--index1', type=int, default=100)
parser.add_argument('--index2', type=int, default=200)
# chatgpt 
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--temperature', type=float, default=0.0)
opt = parser.parse_args()

if not torch.cuda.is_available():
    opt.device = 'cpu'

