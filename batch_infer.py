import pickle
import os
import subprocess
import sys

if __name__ == '__main__':
    dir_id = '1'
    with open(f'tools/{dir_id}_info.pkl', 'rb') as f:
        # 读取文件内容并反序列化为Python对象
        data_loaded = pickle.load(f)
    command_list = data_loaded['sub_command']

    p_list = []

    for cmd in command_list:
        cmd = ['/data2/anaconda3/envs/freedreamer2/bin/python', '-u', '/data2/zhf/workplace/ProPainter/inference_propainter.py', '--video=/data9/zhf/data/xyrs/1_sub/frames', '--mask=/data9/zhf/data/xyrs/1_mask', '--fp16', '--output=/data9/zhf/data/xyrs/1_sub', '--resize_ratio=0.6', '--mask_dilation=12', '--save_frames', '--use_frames=[1469, 1536]']
        print(cmd)
        p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
        p.wait()
        break