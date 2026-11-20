# 分片影像重采样与上游梯度归属

`samples/capture.json` 是离线合成的交接采集，包含原始数据和接收顺序。[交接协议](交接协议.md) 描述各字段及上下游共同使用的数值口径。

`resampling.py` 实现协议的前向重采样与匹配伴随反传：按 tile/request/gradient 事件流工作，不物化整图，滤波与伴随临时张量受 `workspace` 元素数硬约束。

环境使用 Python 3.11 和锁定的 NumPy。首次在本地准备虚拟环境后运行以下命令。

```bash
python -m pip install -r requirements.txt
python -m read_capture samples/capture.json
python -m resampling samples/capture.json
python -m unittest discover -s tests -v
```

## 接口

```python
from resampling import HandoffEngine, replay

engine = HandoffEngine(shape, owners, affine, radius, workspace)
engine.ingest_tile("a", revision=1, observations=[[y, x, value], ...])  # value=None 为作废
view = engine.request("p0", revision_cap=1, centers=[[y, x], ...])     # 前向 + 不可变票据
response = engine.apply_gradient("g0", "p0", gradients)                 # 伴随散射，按行归属
engine.owner_chunks()                                                   # 各拥有者最终稀疏分块
```

也可以用 `replay(capture)` 按采集文件的原始事件顺序一次性回放，拿到引擎与每个事件的返回值。

### 前向返回值（每格点）

| 字段 | 含义 |
| --- | --- |
| `outcome` | `trainable` / `preview_missing` / `missing_no_value` / `no_observation` / `blocked_conflict` |
| `value` | 按剩余有效权重重归一化的 FP64 值；无值时为 `null` |
| `covered_pixels` / `valid_pixels` / `void_pixels` / `missing_pixels` | 钳位合并后的唯一像素身份与分类 |
| `conflicts` | 同版本异值的坐标、修订号与竞争值，受影响的新格点阻塞 |

### 关键语义

* 坐标 `[y,x]`、像素中心在整数坐标，`affine` 为输出中心到源中心的 2×3 映射；两轴独立三角窗，图外格点钳到最近合法格点并按全局坐标合并权重，零权重边界不计覆盖。
* 实测零亮度是合法观测；`null` 作废观测剔除后可重归一化；无收件坐标为缺报，只允许预览，有权重的格点全缺时返回无值。
* 每像素取不超过可见上限的最高修订号；票据只引用签发当次实际读到的版本、有效像素与归一化权重，后到数据不会重新解释旧票据，冲突只阻塞新格点。
* 反传沿票据的归一化权重做线性伴随，按源像素行对应的 `owners` 归属；同一梯度批号重复送达直接返回首次结果且不再累加，载荷不一致会报错；不可训练格点逐条返回拒绝信息。
* 所有正式值与伴随累加使用 FP64（参考比较 `atol=rtol=1e-6`）；`workspace` 只统计滤波与伴随临时张量，逐格点处理并在窗口过大时自动退化为标量路径，输入台账与各拥有者的最终分块不计入。
