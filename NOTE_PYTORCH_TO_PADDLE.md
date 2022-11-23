# Note: Converting PyTorch Code to PaddlePaddle Platform

Here is a note for converting PyTorch to PaddlePaddle:

In most cases, one just replace `torch` by `paddle`:

|              PyTorch              |            PaddlePaddle            |
|:---------------------------------:|:----------------------------------:|
|          `import torch`           |          `import paddle`           |
|      `import torch.nn as nn`      |      `import paddle.nn as nn`      |
| `import torch.nn.functional as F` | `import paddle.nn.functional as F` |

Some cases one must notice (`inp` and `out` are `torch.Tensor` or `paddle.Tensor`):

|                            Description                             |                            PyTorch                            |                                    PaddlePaddle                                     |
|:------------------------------------------------------------------:|:-------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|
|           paddle's shape parameter must be a tuple/list            |                 `out = inp.view(b, c, h, w)`                  |                          `out = inp.reshape((b, c, h, w))`                          |
|                        different transpose                         |                  `out = inp.transpose(1, 2)`                  |                          `out = inp.transpose((0, 1, 2))`                           |
| pytorch's max returns a tuple, while paddle returns the max values |                     `out = inp.max(0)[0]`                     |                                 `out = inp.max(0)`                                  |
| pytorch's min returns a tuple, while paddle returns the min values |                     `out = inp.min(0)[0]`                     |                                 `out = inp.min(0)`                                  |
|                  paddle has no inplace operations                  |                      `out.unsqueeze_(0)`                      |                              `out = inp.unsqueeze(0)`                               |
|                        different api names                         |                    `optimizer.zero_grad()`                    |                              `optimizer.clear_grad()`                               |
|                       different input shape                        |                  `cross_entropy(pred, mask)`                  | `cross_entropy(pred.transpose((0, 2, 3, 1)), mask)` (`pred` has shape [B, C, H, W]) |
|       paddle's linear weights has **transposed** dimensions        |     `nn.Linear(2, 6).weight.shape == torch.Size([6, 2])`      |                      `nn.Linear(2, 6).weight.shape == [2, 6]`                       |
|         paddle's conv weights has **the same** dimensions          | `nn.Conv2d(2, 6, 3).weight.shape == torch.Size([6, 2, 3, 3])` |                  `nn.Conv2D(2, 6, 3).weight.shape == [6, 2, 3, 3]`                  |
