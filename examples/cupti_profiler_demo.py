# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Measuring CUPTI performance metrics using CUPTI Profiler.
This is supported on V100 and higher NVIDIA GPUs.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class Xor(nn.Module):
    def __init__(self):
        super(Xor, self).__init__()
        self.fc1 = nn.Linear(2, 3, True)
        self.fc2 = nn.Linear(3, 1, True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x


def train_xor():
    xor = Xor()
    xor.to("cuda:0")
    inputs = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = torch.Tensor([0, 1, 1, 0]).view(-1, 1)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(xor.parameters(), lr=0.01)
    xor.train()

    # Configure CUPTI profiler
    with torch.profiler.profile(
        # Currently only supports events on CUDA/GPU side, so do not add ProfilerActivity.CPU
        # this restriction is being removed
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        experimental_config=torch.profiler._ExperimentalConfig(
            profiler_metrics=[
                # Metrics can be picked from
                # https://docs.nvidia.com/cupti/r_main.html#r_profiler
                # or use kineto__tensor_core_insts, kineto__cuda_core_flops
                "kineto__tensor_core_insts",
                "dram__bytes_read.sum",
                "dram__bytes_write.sum",
            ],
            profiler_measure_per_kernel=False,
        ),
    ) as prof:
        for input, target in zip(inputs, targets):
            input, target = input.to(device="cuda:0"), target.to(device="cuda:0")
            optimizer.zero_grad()
            output = xor(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

    prof.export_chrome_trace("/tmp/example_trace_cupti.json.gz")

    for idx in range(0, 15001):
        for input, target in zip(inputs, targets):
            input, target = input.to(device="cuda:0"), target.to(device="cuda:0")
            optimizer.zero_grad()
            output = xor(input)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if idx % 5000 == 0:
            print(f"Epoch {idx} Loss: {loss.data.cpu().numpy()}")


train_xor()
