# Elastic Parameter Server

## Overview


ElasticDL通过对深度学习任务进行合理调度，满足用户需求，同时提高集群利用率。

深度学习任务的特点：

- 推荐业务使用CPU集群，PS架构，async训练
- CV/NLP业务使用GPU集群，AllReduce架构，bsp训练

用户需求：

- 优先级高的任务尽快满足，资源不够的情况下，可以先运行起来；后续资源充沛时，可以动态扩展。
- 优先级低的任务减少资源，甚至停止。

集群资源情况：

- CPU： 相对充裕，一般管够
- memory：相对充裕
- disk：相对充裕，可以认为是无限量
- GPU：相对紧缺，成本高，CV/NLP任务一般独占
- network bandwidth：相对紧缺，深度学习任务的模型参数通信需要大量的网络带宽（千兆网--> RDMA --> NVlink，可以看到对网络带宽的极度需求）

结合上述背景，我们提出一下解决思路：

- CV/NLP任务采用支持动态伸缩的AllReduce架构来解决。CV/NLP任务使用GPU训练，网络参数可以全部放置在一张GPU中。换句话说，只要一个GPU卡“存活”，我们都可以从这张卡开始恢复，进行动态伸缩。

- 推荐任务采用Parameter server架构来解决。推荐任务通常需要一个大的embeddig table，超过单机内存；并且训练数据量巨大，通常使用async SGD来训练。这导致parameter server架构有两种角色，存储全局参数的pserver和训练数据的worker。


## ElasticDL of parameter server architecture

这里，对parameter server架构下，支持容错和动态伸缩进行详细分析。

关于容错，这里我们需要首先讨论为什么pserver/worker pod会挂掉，原因有如下几个可能：

- 硬件偶发性出现故障，直接挂掉
- 当前深度学习任务的优先级较低，被高优先级的任务抢占，导致pod被kill掉

第一种情况是偶发的，而且parameter server架构占用的资源是CPU/memory/网络带宽，这三个资源一般也不容易出现硬件故障。

第二种情况我们这里详细分析一下，对worker和pserver进行分类讨论：

### Worker

- 资源需求：计算密集型，需要的是CPU/network bandwidth，对CPU的需求更大
- 容错：对于worker而言，在async SGD的情况下，worker可以随便挂掉的。我们可以给worker pod上标记prority，低优先级的worker pod会因被高优先级的pod抢占而被kill掉
- 动态伸缩：同时worker属于CPU密集型，动态伸缩也很容易，多一些CPU的资源，就可以多发起几个worker


### Pserver

- 资源需求：network bandwidth/memory/CPU（少量，执行optimize计算），由于采用多个pserver共同存储一个全局模型，单个pserver对memory的需求也不大。重点是network bandwidth，需求量极大，需要满足多个worker发起频繁的push/pull请求
- 容错：由于pserver占用的主要硬件资源是network bandwidth，而memory/CPU又相对充裕，我们能不能把对pserver的容错合并到动态伸缩一起考虑呢？换句话说，对于偶发性的硬件机器故障，pserver支持从checkpoint加载，从而重新发起任务。对于常见的高优先级任务抢占，我们采取限制低优先级pserver network bandwidth的方式，来减少低优先级pserver的资源占用，而不是选择kill掉pserver pod。低优先级pserver pod占用的memory/CPU可以相对忽略不计，我们不做处理
- 动态伸缩：我们维持pserver pod的数目不变，始终保持跟parameter sharding策略中产生的shard数目保持一致。在资源分配上，我们调整相应pserver pod的网络带宽即可完成对pserver的动态伸缩

## Summary

因为worker是无状态的，处理相对简单；pserver是有状态的，处理相对复杂。但是我们考虑到pserver对资源占用大头是network bandwidth，而CPU/memory相对充裕，我们把对pserver的容错和动态伸调整合在一起考虑，转化为限制pserver的network bandwidth。