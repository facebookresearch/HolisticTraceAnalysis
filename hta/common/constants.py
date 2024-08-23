# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The Max queue size is not really documented and is an implementation detail
# https://forums.developer.nvidia.com/t/maximum-number-of-operations-in-a-stream/255260
# We anecdotally see 1022 - 1024 to cause blocking of runtime operations.
CUDA_MAX_LAUNCH_QUEUE_PER_STREAM: int = 1024
