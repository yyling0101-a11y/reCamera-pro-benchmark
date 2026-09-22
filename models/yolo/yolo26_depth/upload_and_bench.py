#!/usr/bin/env python3
import paramiko
from scp import SCPClient
import os

HOST = '192.168.3.207'
USER = 'root'
PASSWD = 'recamera.1'
LOCAL_DIR = '/home/seeed/recamera_pro/benchmark/models/yolo/yolo26_depth'
REMOTE_DIR = '/userdata/benchmark/models/yolo/yolo26_depth'

# Upload
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASSWD, timeout=30)
scp = SCPClient(ssh.get_transport())

ssh.exec_command(f'mkdir -p {REMOTE_DIR}')
scp.put('yolo26n-depth_640x640_W8A8.rknn', f'{REMOTE_DIR}/')
scp.put('bus.jpg', f'{REMOTE_DIR}/')
print("Upload complete")

# Quick benchmark
cmd = f'''
cd {REMOTE_DIR} && python3 -c "
import time
import numpy as np
from rknnlite.api import RKNNLite

rknn = RKNNLite()
ret = rknn.load_rknn('yolo26n-depth_640x640_W8A8.rknn')
if ret != 0:
    print(f'Load failed: {{ret}}')
    exit(1)

ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
if ret != 0:
    print(f'Init failed: {{ret}}')
    exit(1)

# Prepare input
import cv2
img = cv2.imread('bus.jpg')
img = cv2.resize(img, (640, 640))
img = img.astype(np.uint8)
img = np.expand_dims(img, axis=0)

# Warmup
print('Warmup...')
for _ in range(20):
    out = rknn.inference(inputs=[img])

# Benchmark
print('Benchmark...')
times = []
for i in range(100):
    t0 = time.perf_counter()
    out = rknn.inference(inputs=[img])
    times.append((time.perf_counter() - t0) * 1000)
    if i == 0:
        print(f'Output shape: {{out[0].shape}}')
        print(f'Output dtype: {{out[0].dtype}}')
        print(f'Output range: [{{out[0].min():.3f}}, {{out[0].max():.3f}}]')

times = np.array(times)
print(f'Avg: {{np.mean(times):.2f}} ms')
print(f'Min: {{np.min(times):.2f}} ms')
print(f'Max: {{np.max(times):.2f}} ms')
print(f'FPS: {{1000/np.mean(times):.1f}}')

rknn.release()
"
'''

stdin, stdout, stderr = ssh.exec_command(cmd, timeout=60)
out = stdout.read().decode('utf-8')
err = stderr.read().decode('utf-8')
print("STDOUT:")
print(out)
if err:
    print("STDERR:")
    print(err[:500])

scp.close()
ssh.close()
