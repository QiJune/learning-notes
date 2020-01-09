## mnist test


CPU device


```bash
docker run -u $(id -u):$(id -g) --rm -v eager_async:/eager_async --runtime=nvidia -it tensorflow/tensorflow:latest-gpu-py3 /bin/bash
cd /eager_async
python mnist_test.py
```

- sync mode: 5.481237s
- async mode: 4.755175999999999s

## resnet50 test


GPU device


```bash
docker run -u $(id -u):$(id -g) --rm -v eager_async:/eager_async --runtime=nvidia -it tensorflow/tensorflow:latest-gpu-py3 /bin/bash
cd /eager_async
python resnet50_test.py
```


- sync mode: 6.446621275s
- async mode: 3.3482721249999994s


## add callback discussion

issue: https://github.com/tensorflow/tensorflow/issues/33274

Thanks for your reply.  In some large model training, such as a model with a big embedding table out of a single computer memory, we usually use parameter server distributed strategy. The gradients are pushed to the parameter servers, and then aggregated and applied to parameters stored in parameter servers.

I notice that there is an Async mode in [EagerExecutor](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/eager/eager_executor.h#L95) in the C++ runtime of TF2.0 Eager version. It launches EagerOperation asynchronously. In addition, there is a `RemoteCopyNode` which supports "remote-->remote"/"remote-->local"/"local-->remote" copy. The node could be launched by EagerExecutor. So, in C++ runtime, this problem is well addressed. The computation and communication are overlapped.

However, we want to achieve such effects in Python without touching or modifying C++ runtime.

One way is to support inserting an end callback function/or operator if possible to a backward operator. Once the backward operator finishes its backward computation, the callback function will be launched. Thus, if we could insert `send_gradient`  or `allreduce_gradient` end callback to a backward operator, the computation and communication will be overlapped simply with Python API.

Following is the logic which gets the gradient function of an operator:
https://github.com/tensorflow/tensorflow/blob/7b80146babce4a998e4654f5f55e09c3e6f1144d/tensorflow/python/eager/backprop.py#L119-L141
Maybe we could add another callback argument here.

I am not sure if I find the right place to add such logic. I am willing to work on this if the callback mechanism is reasonable.
