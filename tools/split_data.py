import cv2
import numpy as np
import os
import pickle
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor


def split_frames(data, dir_id):
    min_buffer = 10
    max_frames = 200
    mask_frames_sub = []
    data = [[line.split('\t')[0], eval(line.split('\t')[1].strip())] for line in data]
    data = sorted(data, key=lambda x: int(x[0].split('.')[0]))
    no_mask_frames = [x[0] for x in data if len(x[1])==0]

    # text = "张哥的手指可真长呀"
    # for ii, x in enumerate(data):
    #     _, cc = x
    #     if len(cc) != 0:
    #         for jj in cc:
    #             if text in jj['transcription']:
    #                 print(ii)
    #                 break

    i = 0
    up_use = False
    while i < len(data):
    # for i, frame in enumerate(data):
        name, text = data[i]
        if len(text) != 0:
            start = i
            end = i
            start_num = 1
            end_num = 1
            while start_num <= min_buffer+1 and start >=0:
                if up_use:
                    start = i-10
                    up_use = False
                    break
                start -= 1
                if len(data[start][1]) == 0:
                    start_num += 1
                else:
                    start_num = 1

            while end_num <= min_buffer+1 and end <= (len(data)-2) and (end - start) < max_frames:
                end += 1
                if len(data[end][1]) == 0:
                    end_num += 1
                else:
                    end_num = 1
            # tmp = [x[0] for x in data[start:end]]
            tmp = [start, end]
            if (end - start) == max_frames:
                up_use = True
            else:
                up_use = False
            if tmp not in mask_frames_sub:
                mask_frames_sub.append(tmp)
            i = end
        i += 1

    default_command = (
            [sys.executable, "-u"]
            + ["/data2/zhf/workplace/ProPainter/inference_propainter.py"]
            + [
                f"--video=/data9/zhf/data/xyrs/{dir_id}_sub/frames",
                f"--mask=/data9/zhf/data/xyrs/{dir_id}_mask",
                f"--output=/data9/zhf/data/xyrs/{dir_id}_sub",
                f"--resize_ratio=0.6",
                f"--mask_dilation=12",
                f"--fp16",
                f"--save_frames"
            ]
    )
    sub_command = []
    for i in mask_frames_sub:
        current_command = default_command.copy()
        start, end = i
        current_command.append(f"--use_frames=[{start}, {end}]")
        sub_command.append(current_command)

    res_data = {
        'no_mask_frames': no_mask_frames,
        'sub_command': sub_command
    }

    with open(f'{dir_id}_info.pkl', 'wb') as f:
        pickle.dump(res_data, f)


if __name__ == '__main__':
    dir_id = '1'
    ocr_res_path = f'/data9/zhf/data/xyrs/{dir_id}_res'
    img_dir = f'/data9/zhf/data/xyrs/{dir_id}'

    ocr_res_prefix = "system_results"
    ocr_res_file_list = [x for x in os.listdir(ocr_res_path) if ocr_res_prefix in x]
    data = []
    for each_file in ocr_res_file_list:
        with open(os.path.join(ocr_res_path, each_file), 'r') as f:
            data.extend(f.readlines())

    split_frames(data, dir_id=dir_id)
