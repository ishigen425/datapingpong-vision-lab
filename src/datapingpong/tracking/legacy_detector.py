import numpy as np

def detect(heatmap, shape, thread=0.5):
    heatmap_thread = np.where(heatmap < 0.5, 0, heatmap)
    return np.unravel_index(np.argmax(heatmap_thread), shape)

def judge(target_heatmap, output_heatmap, shape, sigma=10):
    max_y, max_x = detect(output_heatmap, shape)
    tar_y, tar_x = detect(target_heatmap, shape)
    dist = np.sqrt((tar_x-max_x) ** 2 + (tar_y-max_y) ** 2)
    if dist > sigma:
        return 0
    return 1
