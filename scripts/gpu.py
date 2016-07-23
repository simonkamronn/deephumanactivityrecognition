import os

num_gpu = 7
cuda_cmd = 'ssh -t -t gpu%d "cuda-smi"'
_format = 'Device  0 [nvidia-smi  3] [PCIe 0000:0a:00.0]:  GeForce GTX TITAN X (CC 5.2):    23 of 12287 MiB Used'

usage = dict()
for gpu in range(1, num_gpu+1):
    # os.system(cuda_cmd % gpu)
    usage[gpu] = os.popen(cuda_cmd % gpu).read()
    print(usage[gpu])
