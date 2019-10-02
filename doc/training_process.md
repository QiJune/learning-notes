# Training Process

## Overview

### Dataset

我们一般把数据集分成三份，分别是：

- training dataset
- validation dataset
- test dataset

training dataset用来训练网络参数；validation dataset用来调整超参；test dataset是最终的效果

### Training 

一般情况下，我们从training dataset中读取一个minibatch的数据进行训练。一个minibatch我们称一个step，或者一个iteration。

当training dataset中的所有数据都遍历一轮，我们称为一个pass，或者一个epoch。

在若干次iteration之后，我们会使用validation dataset中的数据，对模型进行评价，这里会有一些evaluation metrics，比如ROC曲线，正确率等。

在一个epoch之后，我们通常会对当前模型进行保存，做一次checkpoint。同时我们也会跑一下validation dataset中的数据，得到当前checkpoint的evaluation metrics。

最后，我们从若干个epoch的checkpoint中，选取一个evaluation metrics最好的作为最终模型。


## Parameter server


Master负责对training process的整体控制。

Worker和Pserver需要接受master的控制。其中worker负责完成training task和evaluation task；pserver提供kv store以及checkpoint服务。

1. master对training dataset进行切分，得到一段数据的<start_addr, end_addr>，然后打包成一个training task发送给worker节点 （worker节点收到training_task，解析得到数据的位置，从这里开始读取，并进一步切分成minibatch数据，进行training任务）
2. master会记录总共打包了多少个minibatch的task出去，当到达设定的iteration之后，会开始打包validation dataset数据，形成许多个evaluation task，发送给worker节点（worker节点开始evaluation任务。到目前为止，worker之间都是相互独立的，即每个worker只关注自己拿到的任务，做完即可。evaluation也只是当前worker对自己拿到的部分validation dataset进行打分）
3. 当master发现所有的training dataset都已经发送完毕之后，master会通知pserver，让所有pserver启动checkpoint任务 （这时，pserver会停止更新操作，对push上来的gradient直接丢弃，并且把当前模型参数写入磁盘）
4. 之后，master会打包validation dataset给worker，要求worker对checkpoint进行打分 （这里要注意，worker需汇报必要中间结果给master，master得到所有worker对所有validation dataset的评价后，进行汇总和计算，最终给出一个分数）
5. master得到一个epoch训练结束后对全部validation dataset的打分，并且把对应的evaluation结果写入磁盘
6. master重新回到step 1，开始下一个epoch的训练


**Note**

每个epoch之内的evaluation是worker针对局部数据做的，仅仅是参考；每个epoch之间的evaluation，是对全部validation dataset进行统计汇总之后得到的结果，用于最终选择模型。




