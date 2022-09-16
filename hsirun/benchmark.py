import argparse
import json
import os
from datetime import datetime
from ast import literal_eval
import platform

import torch
import torch.cuda
import torch.backends.cudnn

from torchlight.utils import instantiate

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True


def main(arch, input_shape, total_steps=10, save_path='benchmark.json'):
    net = instantiate(arch)

    net = net.cuda().eval()
    input = torch.randn(*input_shape).cuda()

    total_params = sum([param.nelement() for param in net.parameters()]) / 1e6
    print("Number of parameter: %.2fM" % (total_params))
    
    # warm up for benchmark
    for _ in range(10):
        net(input)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    steps = int(total_steps)
    start.record()

    for _ in range(steps):
        net(input)
    end.record()

    torch.cuda.synchronize()

    avg_time = start.elapsed_time(end) / steps
    print('Time: {} ms'.format(avg_time))

    database = {}
    if os.path.exists(save_path):
        database = json.load(open(save_path, 'r'))
    entry = database.get(arch, [])
    dtstr = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    entry.append({
        'runtime': avg_time, 'params': total_params, 'date': dtstr,
        'os': platform.platform(), 
        'processor': platform.processor(),
    })
    database[arch] = entry
    with open(save_path, 'w') as f:
        json.dump(database, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Benchmark model runtime and params')
    parser.add_argument('-a', '--arch', type=str, required=True)
    parser.add_argument('-t', '--steps', type=int, default=10)
    parser.add_argument('-s', '--save-path', type=str, default='benchmark.json')
    parser.add_argument('-i', '--input-shape', type=str, default='[1,1,31,512,512]')
    args = parser.parse_args()
    input_shape = literal_eval(args.input_shape)
    main(args.arch, input_shape, args.steps, args.save_path)
