# parameter server

## parameter sharding

多个pserver，然后参数按照一定的规则shard在pserver上。

对于embedding table可以理解为一个key value的结构，key就item id，value就是对应id的embedding vector。同时key跟pserver的id之间有一个hash的映射关系，给定一个key，就能查到对应的pserver id，也就意味着这个embedding vector实际上存放在该pserver上。

对于其他的parameter，我们设置一个size，如果该parameter大于这个size，那么这个parameter也需要被sharding。这样的parameter通常是一个固定大小的tensor，我们可以按照把tensor按行切分为pserver的数目，即切分为若干个subtensor。我们给定一个parameter的名字，加上对应subtensor的偏移，就可以唯一确定一个key。同样这个key也可以通过一个hash函数映射到一个pserver id，该pserver上就存放着对应的subtensor。

综上所述，我们有需要一个kvstore，这个kvstore的key是一个int，value是可能是一个vector，也可能是一个二维tensor。

## init parameter

embedding table参数的初始化是lazy的，在训练中fetch参数的时候一起初始化。

其他parameter的初始化可以在训练之前进行。

## pull parameter

对于每一个worker，在每个layer进行forward计算之前，需要从pserver上拉去parameter到本地。

对于embedding layer来说，输入是一个batch的数据，因此item id是一个vector。然后需要把这个item id的vector进行拆分，拆分后的小组都对应着一个pserver id。因此，这里会对若干个pserver发起请求，从pserver那里拿到对应的embedding vector。

这里有一个处理，对于online learning来说，随时可能有新的id出现，那么pserver上面没有该item id对应的embedding vector，那么这时，pserver应该创建一个新的embedding vector，并且初始化，然后再返回给worker。

对于其他的parameter来说，如果没有被切分，那么发起请求，从某一个pserver上拉去parameter。如果被切分，那么同样，从若干个pserver上请求若干个subtensor到本地。这里要注意，从一个tensor得到subtensor，实际上是zero-copy的，我们只需要做一个view就可以。

pull parameter跟该layer的forward计算是串行的，我们只能pull参数到worker本地之后，才能开始forward计算。

## push gradient

在worker进行一层的backward计算之后，会产生gradient，这时我们需要把gradient push到pserver上。

push gradient与pull parameter的逻辑类似。

对于embedding来说，同样拆分item id的vector，对本地的gradient分成若干组，push到pserver。这里要注意，由于一个batch的训练数据中，可能存在若干个相同的id，那么这里的gradient中就会存在对同一个id的若干个gradient vector。这里要把这些相同id的若干个gradient vector做一个加法处理，目的是消除重复。

对于其他的parameter，相同的处理push到pserver。

这里发起push gradient是非阻塞的，发起操作之后，并不影响下一层backward的计算。

## optimize parameter

pserver上会执行一定的优化算法，把gradient更新到parameter上。这里会涉及到不同策略组合，一个维度是bsp/async的更新方式，另外一个维度是不同的优化算法

- 更新方式： bsp/async/ssp
- 优化算法：sgd/adam/momentum

下面针对不同的策略组合描述详细过程：

- bsp-sgd: pserver得到所有worker传来的gradient之后，进行聚合求平均的操作，然后再更新到parameter上。这里要注意，聚合的时候同一个key对应的vector或者tensor要做加法。
- async-sgd：pserver只要得到一个worker传来的gradient，就直接进行更新操作。

## other questions

### bsp模式下跨batch

当训练任务跨越两个batch时，需要考虑上一个batch的gradient正在/已经push到pserver，pserver未开始/正在对parameter进行优化，下一个batch的训练已经开始，需要从pserver上pull参数。

这里实际上有一个依赖关系需要保证，pull下来的参数一定是已经被上一个batch数据训练优化过的。

如何保证？

### async模式下读写锁

async模式下是否需要读写锁？

最终的目标是收敛精度和收敛速度，读写锁一方面加了锁，确实会影响每一次pull/update的速度；但是另外一方面，计算数值更加稳定，如果能保证收敛精度以及收敛速度，那么其实是有收益的。

这里需要做实验试一试？
