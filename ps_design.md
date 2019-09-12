# Parameter Server Design

## Assumption

- 只支持async更新
- pserver的动态增减暂不考虑

## Overview

总共有三种角色：

- master
- pserver
- worker

下面会详细描述。


## Master

- 负责创建，删除，调度，监控pserver/worker
- 定义模型中的parameter的sharding策略，给每个parameter(或者切分后的parameter)产生一个key
- 定义parameter key到pserver id的映射策略

工作流程

1. 根据用户配置，创建相应数目的pserver和worker
2. 根据用户提供的模型，产生parameter sharding策略
3. 初始化parameter，同步到pserver中
4. 启动pserver/worker，开始训练
5. 监控集群状态，可以根据集群空满状态，以及任务优先级，选择动态的增减worker数目


## Pserver

- 提供一个kv store的服务，可以pull/push <key, value>到这里
- 提供optimizer，可以把拿到的gradient进行优化，再写入到kv store中


工作流程

1. 等待worker push <key, gradient>到pserver
2. pserver查询kv store，得到对应的<key, parameter>
3. pserver调用optimizer部分，把<key, gradient> apply到<key, parameter>上
4. 把更新后的<key, parameter>写会kv store

pserver的kv store的服务可以是自己实现的，也可以用已有方案，比如redis。目前想法是一个pserver有自己独立的


## Worker

- 定义一个forwad-backwad的计算过程
- 定义一个data loader模块

工作流程

1. 在每一层forward layer计算之前，发送<key>到pserver中，把<key, parameter>给pull下来
2. 进行当前forward layer计算
3. 进行backward layer计算，产生<key, gradient>
4. 在每一层backward layer计算之后，把产生的<key, gradient> push到pserver