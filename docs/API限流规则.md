# 币安 API 限流规则

> 调研日期：2026-09-20（lrt-pc，代理 7897，实时读响应头 `x-mbx-used-weight-1m` 实测 + 官方文档核对）。
> 适用：`data/binance.py`、`live/` 轮询、历史回填。

## 1. 限流模型

- 限流单位是 **weight**（不是请求数），窗口固定 **1 分钟**，按 **出口 IP** 计。
- 每次响应头带 `x-mbx-used-weight-1m`（本 IP 当前分钟已用 weight），可实时自测。
- 超限后果：
  - HTTP `429 Too Many Requests` → 该 IP 封禁 **120 秒**；
  - 120 秒内再次超限 → HTTP `418` → 封禁 **60 分钟**。
- Spot 与合约是**两套独立限额**（不同域名、不同计数器）。

## 2. IP 限额

| 市场 | 端点域 | 限额 | 来源 |
|---|---|---|---|
| Spot | `api.binance.com` | **6000 weight / 分钟** | 官方 FAQ + 实测 header |
| USDⓈ-M 合约 | `fapi.binance.com` | **2400 weight / 分钟** | `/fapi/v1/exchangeInfo` 的 `rateLimits` 字段（interval=MINUTE, limit=2400），实测确认 |
| 合约下单 | fapi | 1200 orders/分钟，300/10 秒 | 文档（`data/binance.py` 头注释同） |
| Spot 下单 | api | 10 次/10 秒，200,000 次/24 小时 | 官方 FAQ |

## 3. 端点 weight 表（本项目用到的）

| 端点 | weight | 备注 |
|---|---|---|
| `/api/v3/ping`、`/fapi/v1/ping`、`/fapi/v1/time` | 1 | 实测 delta≈1 |
| `/fapi/v1/exchangeInfo`、`/api/v3/exchangeInfo` | 1~数 | 全量响应大，实测 delta 含噪声 |
| **klines 系列**（spot/合约，分档） | limit 1–100 → **1**；101–500 → **2**；501–1000 → **5**；>1000 → **10** | 官方分档，本地实测 limit=1000 delta=5、limit=1500 delta=10，吻合 |
| `/fapi/v1/ticker/24hr`（不传 symbol，全市场） | **~80** | 单次拿全部合约，实测 delta 76–88 |
| `/fapi/v1/fundingRate`（单 symbol） | 1（limit>1000 时 10） | 分档同 klines |
| `/fapi/v1/premiumIndex`（不传 symbol，全市场） | 10 | 文档值；本地 delta 受窗口重置污染未测净 |
| `/fapi/v1/openInterest`（单 symbol） | 1 | 文档 |

换算参考：合约 2400/分钟 ÷ 10 = **每分钟最多 ~240 页 klines（1500 根/页）**；全市场 ticker/24hr 一次 ≈ 80 weight ≈ 每分钟可做 ~30 次。

## 4. 本地实测方法（可复现）

```bash
P=http://127.0.0.1:7897
w() { curl -s --max-time 15 -x $P -o /dev/null -D - "$1" \
       | grep -i 'x-mbx-used-weight-1m' | tr -d '\r ' | cut -d: -f2; }
a=$(w "https://fapi.binance.com/fapi/v1/ping"); sleep 0.3
b=$(w "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=1000")
echo "delta=$((b-a))"   # 期望 5
```

## 5. 实测中踩到的坑（重要）

1. **同一出口 IP 有其他流量**：测时基线 weight 在 ~400–770 间漂移，说明该出口 IP 上还有别的程序在打币安。
   → 差值法测单端点 weight 会被污染；klines 分档因增量小（5/10）仍可靠，重端点（80+）只能看趋势。
   → **对生产有实际含义**：`soft_limit_ratio: 0.80` 判断的是"本进程记录的已用 weight"，不感知同 IP 其他流量；
     若共享出口，真实窗口余量比代码认为的少。VPS 独占 IP 时问题不大，家用宽带多服务共享时需留意 429 频率。
2. **分钟窗口边界**：`x-mbx-used-weight-1m` 在窗口滚动瞬间骤降，两次采样跨分钟会得到**负差值**（实测出现过 delta≈-600）。
   → 自动化测重/监控必须处理窗口重置（负差值视为重置，重采一次）。
3. **curl header 带 `\r`**：`grep` 出的值带回车，bash 算术直接报 `invalid arithmetic operator`。须 `tr -d '\r'`。
4. **`fundingInfo` 只返回非默认周期合约**（与限流无关但易误判为空列表，`data/funding.py` 已注释）。

## 6. 减少请求数的策略（收益排序）

1. **不传 symbol 拉全市场**：`ticker/24hr`、`premiumIndex` 单请求 = 全部合约数据（1 次 ~80/10 weight 替代 N 次单币请求）。
   项目已采用（`live/strategy.py` 全市场成交额走单次 ticker）。
2. **K 线吃满 limit=1500**（weight 10/1500 根）：10 根/页翻 150 次（weight 150）vs 1500/页翻 1 次（weight 10），省 15 倍。
   `binance.py` 已用 `KLINES_MAX_LIMIT=1500` 分页。
3. **WebSocket 替代 REST 轮询**：实时行情/资金费率走 `wss` 流，REST 只做历史回填，实时路径 0 请求 weight。
4. **分层拉取**：全市场只拉轻量端点（ticker/24hr），历史 K 线/费率只对通过筛选的候选币拉。
5. **缓存**：`_cached_get` + `klines_closed` TTL，已实现。

## 7. 本项目现有防护（`config/config.yaml` → data.rate_limit）

| 配置 | 值 | 作用 |
|---|---|---|
| `futures_weight_per_min` | 2400 | 合约硬限额 |
| `spot_weight_per_min` | 6000 | Spot 硬限额 |
| `soft_limit_ratio` | 0.80 | 达 80% 主动休眠到下一分钟窗口 |
| `base_backoff_seconds` / `max_backoff_seconds` | 0.5 / — | 429 后指数退避 |
| `max_retries` | — | 重试上限 |

`data/binance.py` 用 `_WeightTracker` 按响应头记账。设计方向正确；已知短板见 §5.1（不感知同 IP 他流量）。

## 8. 参考

- Spot 限流 FAQ：https://www.binance.com/en/support/faq/detail/360004492232
- Spot REST 通用（IP Limits 说明）：https://developers.binance.com/en/docs/products/spot/rest-api
- 合约市场数据端点 weight 表：https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data
