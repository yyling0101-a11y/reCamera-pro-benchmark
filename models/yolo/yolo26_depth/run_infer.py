#!/usr/bin/env python3
import paramiko
from scp import SCPClient

HOST = '192.168.3.207'
USER = 'root'
PASSWD = 'recamera.1'
REMOTE_DIR = '/userdata/benchmark/models/yolo/yolo26_depth'
LOCAL_DIR = '/home/seeed/recamera_pro/benchmark/models/yolo/yolo26_depth'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASSWD, timeout=30)
scp = SCPClient(ssh.get_transport())

# Upload inference script
scp.put('infer.py', f'{REMOTE_DIR}/')
print("Uploaded infer.py")

# Run inference
cmd = f'''
cd {REMOTE_DIR} && python3 infer.py \\
    --model yolo26n-depth_640x640_W8A8.rknn \\
    --image bus.jpg \\
    --output result_bus.jpg \\
    --imgsz 640 \\
    --max-depth 15.0 \\
    --runs 100 \\
    --warmup 20
'''

print("Running inference...")
stdin, stdout, stderr = ssh.exec_command(cmd, timeout=120)
out = stdout.read().decode('utf-8')
err = stderr.read().decode('utf-8')
print("STDOUT:")
print(out)
if err:
    print("STDERR:")
    print(err[:800])

# Download results
print("\nDownloading results...")
scp.get(f'{REMOTE_DIR}/result_bus.jpg', f'{LOCAL_DIR}/result_bus.jpg')
print(f"Downloaded: result_bus.jpg")

scp.get(f'{REMOTE_DIR}/result_bus_depth.jpg', f'{LOCAL_DIR}/result_bus_depth.jpg')
print(f"Downloaded: result_bus_depth.jpg")

scp.close()
ssh.close()
print("Done!")
