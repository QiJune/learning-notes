# Parameter Server Failover


## Assumption

- checkpoint在分布式文件系统里面，理论上文件系统是不会挂掉的
- master是一个训练任务的中心，理论上只要master不挂掉，就可以重启整个训练任务


## Analysis


- pserver占用的是内存和网络带宽资源，以及少量的CPU（用来optimize参数）。抢占pserver的核心是减少其网络带宽,我们认为内存和CPU是相对充裕的
- worker占用的是计算资源，内存和网络带宽。在async模式下，worker的动态伸缩是很容易解决的，也不需要展开讨论


## Solution

### Pserver Failover


对于pserver而言，我们能不能还有一个思路的转换，我们考虑不是减少pserver pod的数目，而是限制pserver pod占用的网络资源。

按照这个思路：

- 把全局的parameter进行sharding处理，得到一个sharding的number，对应这个number创建若干个pserver的pod(在创建pod的时候，会限制这个pod的内存占用，CPU，以及网络带宽)
- 抢占pserver的资源时，把所有的pserver pod给kill掉，然后在新的资源限制下，从checkpoint加载，创建等数目的pserver pod


### Worker Failover

在async SGD更新方式下，相对简单，动弹增减对整体系统没有影响。


## Scenario

- Pserver突然挂掉： master会定时监控pserver的数目，如果发现减少，则会利用k8s重启pserver，需要从checkpoint中加载对应的parameter sharding
- Pserver被抢占1：master收到通知，pserver占用的资源需要减少，则从checkpoint中加载数据创建新的pserver，并且把之前的pserver给kill掉
- Pserver被抢占2：这里可能可以进一步的优化，不是从分布式文件系统中的checkpoint加载，而是从之前pserver内存中的parameter sharding直接进行拷贝，创建新的pserver
