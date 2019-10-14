## mnist test


CPU device


```bash
docker run -u $(id -u):$(id -g) --rm -v eager_async:/eager_async --runtime=nvidia -it tensorflow/tensorflow:latest-gpu-py3 /bin/bash
cd /eager_sync
python mnist_test.py
```

- sync mode: 5.481237s
- async mode: 4.755175999999999s

## resnet50 test


GPU device


```bash
docker run -u $(id -u):$(id -g) --rm -v eager_async:/eager_async --runtime=nvidia -it tensorflow/tensorflow:latest-gpu-py3 /bin/bash
cd /eager_sync
python resnet50_test.py
```


- sync mode: 6.446621275s
- async model: 3.3482721249999994s

