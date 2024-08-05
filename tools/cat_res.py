import cv2
import numpy as np
import os
from multiprocessing import Pool


def process_image(name, ocr_res_path, frames_res, save_dir):
    ocr_img = cv2.imread(os.path.join(ocr_res_path, name))
    if ocr_img is None:
        print(f"Warning: Unable to load {os.path.join(ocr_res_path, name)}")
        return
    ocr_img = ocr_img[:, :int(ocr_img.shape[1] / 2)]

    pad_img = cv2.imread(os.path.join(frames_res, name))
    if pad_img is None:
        print(f"Warning: Unable to load {os.path.join(frames_res, name)}")
        return
    pad_img = cv2.resize(pad_img, ocr_img.shape[:2][::-1])

    res_img = np.hstack((ocr_img, pad_img))
    cv2.imwrite(os.path.join(save_dir, name), res_img)


if __name__ == '__main__':
    dir_id = '101'
    ocr_res_path = f'/data9/zhf/data/xyrs/{dir_id}_res'
    frames_res = f"/data9/zhf/data/xyrs/{dir_id}_sub/frames"
    save_dir = f"/data9/zhf/data/xyrs/{dir_id}_sub/frames_cat"
    os.makedirs(save_dir, exist_ok=True)

    file_name_list = os.listdir(frames_res)

    # 使用多进程池
    with Pool(processes=14) as pool:  # 可以根据需要调整进程数
        pool.starmap(process_image, [(name, ocr_res_path, frames_res, save_dir) for name in file_name_list])

    print("所有图像处理完成。")