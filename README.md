## 我在使用 pytorch 过程中遇到的问题
-------------------------
> 本仓库主要记录自己使用 pytorch 进行深度学习过程中遇到的一些问题和解决办法  
> ㊙ 这些解决方法未必是最佳方案

------------------
------------------

### 1️⃣使用 DataParallel 时,GPU 显存负载不均衡  
- 使用单机多显卡时，可能每张显卡的显存大小不是一致的。而 DataParallel 默认是根据显卡的数量对 batchsize 进行均分，每张显卡都会被分配到一样大小的显存消耗。
修改 `torch > nn > Parallel > scatter_gather.py` 的 `scatter_map` 函数如下  
```python
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            batch_size = obj.shape[0]
            num1 = 2*batch_size//3
            # raise "$$$$$$$$$$"
            # return Scatter.apply(target_gpus, None, dim, obj)
            return Scatter.apply(target_gpus, [num1, batch_size-num1], dim, obj)
   ...
```
