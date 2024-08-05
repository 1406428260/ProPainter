import cv2
import numpy as np
import os
from multiprocessing import Pool


def process_image(args):
    img_name, info, img_dir, save_mask_dir = args
    img = cv2.imread(os.path.join(img_dir, img_name))
    if img is None:
        print(f"Failed to load image: {img_name}")
        return
    mask_img = np.zeros_like(img)
    for each in info:
        bbox = each["points"]
        bbox = np.array(bbox, dtype=np.int32)  # 确保数据类型正确
        bbox = bbox.reshape((-1, 1, 2))  # 调整形状以匹配cv2.fillPoly的要求
        cv2.fillPoly(mask_img, [bbox], color=(255, 255, 255))
    cv2.imwrite(os.path.join(save_mask_dir, img_name), mask_img)


def get_mask(data, img_dir, save_mask_dir, processes=5):
    # 准备数据，每个元素是一个元组：(img_name, info, img_dir, save_mask_dir)
    args_list = [(line.split('\t')[0], eval(line.split('\t')[1].strip()), img_dir, save_mask_dir) for line in data]

    # 创建进程池
    with Pool(processes=processes) as pool:
        pool.map(process_image, args_list)


if __name__ == '__main__':
    dir_id = '101'
    ocr_res_path = f'/data9/zhf/data/xyrs/{dir_id}_res'
    img_dir = f'/data9/zhf/data/xyrs/{dir_id}'
    save_mask_dir = f'/data9/zhf/data/xyrs/{dir_id}_mask'
    ocr_res_prefix = "system_results"
    processes = 10

    os.makedirs(save_mask_dir, exist_ok=True)
    ocr_res_file_list = [x for x in os.listdir(ocr_res_path) if ocr_res_prefix in x]
    data = []
    for each_file in ocr_res_file_list:
        with open(os.path.join(ocr_res_path, each_file), 'r') as f:
            data.extend(f.readlines())

    get_mask(data, img_dir, save_mask_dir, processes)