import torch
import datetime
import numpy as np
import time
start_time = time.time()
end_time = time.time()
print(32//8)
frames = []
target_frame_np = np.random.rand(3, 256, 256)
frames.append(target_frame_np)
frames.append(target_frame_np)
frames_np = np.array(frames)
print(f'frames_np.shape = {frames_np.shape}')
