# CoinTrader

币安加密货币自动化交易研究框架。当前聚焦一个策略：**资金费率套利（funding rate carry）**。

## 这个项目要回答的问题

在扣除真实成本之后，资金费套利还有净收益吗？

在整个框架跑完之前，不写任何执行层代码。框架存在的意义是**证伪**，不是确认。
如果回测显示扣费后收益微薄，正确的结论是「不值得做」。

## 快速开始

```bash
# 1. 环境（需要 uv；按 .python-version 自动装 3.12，按 uv.lock 精确还原依赖）
uv sync

# 2. 自检（不需要密钥、不需要联网）
uv run cointrader doctor

# 3. 跑测试
uv run pytest

# 4. 探活币安接口（需要联网）
uv run cointrader ping

# 5. 科学策略组合回测：3 天平均日成交额、30 期出场均值、0.15% 基准滑点
uv run cointrader backtest --max-symbols 100 --json reports/portfolio.json

# 报告默认写入 reports/backtest/
# 可指定报告目录
uv run cointrader backtest --report-dir reports/backtest-365

# 6. 指定候选池的组合诊断
uv run cointrader portfolio BTCUSDT ETHUSDT
```

## 回测时序约束

回测默认使用 `backtest.rolling_window_periods: 30` 的固定历史预热：窗口未满不交易；
实际资金费率判断使用最近 10 期滑动平均值。信号产生后至少延迟
`execution_lag_bars` 期执行，执行前信号失效会撤单。
默认回测数据范围是最近 365 天（`backtest.history_days`），并非无限历史。

默认自动发现可交易的 USDT 本位永续，按成交额过滤候选，
再在每个资金费事件只使用当时可见的历史数据，最近 10 期平均资金费用于排名和条件判断；
`rolling_window_periods` 仍作为 30 期历史预热窗口，最多持有
`max_positions` 个币，每个仓位 `per_position_weight`。当前默认是最多 3 笔、每笔 33%。
也可以用 `portfolio BTCUSDT ETHUSDT` 限定候选池；它仍然动态选币，不是固定持有这两个币。

回测先把不同结算周期的资金费事件聚合到统一 8h 时间轴。每个 8h 桶只包含
该桶结束时已经发生的事件，缺失桶补 0；所有交易次数、持仓期数和年化都按
8h（每年 1095 期）计算。

选币时使用历史 4h K 线最近 18 根（72 小时）的滚动平均日成交额：滚动 3 天
`quote_volume` 总额除以 3。在时点 `t` 只使用截至 `t` 已经闭合的 K 线，
不会用当前 `futures_tickers_24h()` 回填一年前的成交量。
资金费滚动均值、连续正负费率、排名也只读取当时及之前的数据。入场使用最近 10 期均值，
出场与替换使用最近 30 期均值；30 期固定窗口仍用于历史预热和因果边界。

回测基准情景每条现货/永续成交腿使用 0.15% 滑点；一笔对冲往返包含 4 条腿，滑点成本约为
名义额的 0.6%，另加手续费。

默认回测还把 `risk.max_basis_adverse_pct` 应用于每笔交易的进出场基差损失。

当前策略默认使用入场最近 10 期、出场最近 30 期资金费率滑动平均值：

- 入场：10 期平均年化资金费率至少 30%
- 出场：30 期平均资金费率转负
- 替换：持仓 60 期内禁止替换；61~120 期需高出当前 100%；超过 120 期需高出当前 70%

信号仍延迟一个周期执行，所有比较只使用事件当时已经可见的数据。

## 安全模型

**默认状态下这个程序没有能力碰你的钱。**

三层隔离：

1. **类型隔离** —— `data/` 层的客户端没有任何下单能力，也没有 API 凭证参数。
   它甚至无法签名请求。
2. **运行时闸门** —— 真实下单需要环境变量精确等于 `YES_I_AM_SURE`
   （写 `1` / `true` / `yes` 都不生效，这是刻意的摩擦），
   且默认走测试网，且 `KILL_SWITCH` 文件不存在。
3. **默认干跑** —— 执行层所有方法默认 `dry_run=True`。

静态约束由 `tests/test_safety.py` 强制，测试失败即构建失败。

**密钥永不落盘。** 只从环境变量读取，且 `SecretStr` 包装保证
`repr()` / `str()` / f-string 都只输出指纹，不输出明文。

详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §4。

## 目录结构

```
docs/          架构设计与操作手册
config/        唯一配置源（不含密钥）
src/cointrader/
  data/        只读币安公开 API（无限流+重试+缓存）
  research/    纯计算：成本模型、绩效指标
  backtest/    确定性回测引擎 + 破产情景分析
  execution/   签名 REST、订单状态机、双腿执行、对账（默认禁用）
  live/        实盘编排：启动预检、主循环、信号→意图（上层）
tests/         含安全静态扫描与前瞻偏差证伪
```

依赖方向严格单向：`data/`、`research/`、`backtest/` 不得 import `execution/`；
只有 `live/` 可以同时调用 `data/`、`research/` 与 `execution/`（`test_safety.py` 静态验证）。

## 实盘执行（分阶段，见 开发设计文档.md）

执行层已实现 M1-M3（签名 REST/限流、Decimal 规则归一化、SQLite 事件账本、
订单状态机、双腿开平仓与 UNKNOWN 恢复、用户数据流、对账器），并带完整单测。
运行模式在 `config.yaml → execution.mode`：

| 模式 | 行为 |
|---|---|
| `paper`（默认） | 不发私有请求；模拟走 backtest/broker 路径 |
| `testnet` | 测试网真实订单（`cointrader live run`） |
| `shadow` | 读真实行情/账户，只生成意图不下单 |
| `live` | 主网真实订单：需 `COINTRADER_TRADING_ENABLED` 魔法串 + `COINTRADER_USE_TESTNET=false` + `--confirm-live` 三重确认 |

常用命令：

```bash
uv run cointrader live doctor   # 启动预检：时钟偏移/凭证/停机开关/模式
uv run cointrader live run      # 主循环（Ctrl-C 优雅停机：关流+释放单实例锁）
```

⚠️ M4（testnet 端到端 7 天）与 M6（主网 canary）尚未通过，
`live run` 目前只应在 testnet 环境使用；主网启动必须走完文档 §13 清单。

## 已知限制

- 币安只保留约 **1 年**资金费历史，回测跨度受此限制
- 资金费结算周期有 1h / 4h / 8h 三种，年化必须按各合约自己的周期折算
- 回测用的是历史平均滑点，极端行情下可能差 10 倍
- 交易所尾部风险无法对冲，只能靠仓位控制（见 `backtest/scenarios.py`）
- 实盘链路未经 testnet 长跑验证（M4 未完成）；主网放量（M7）在此之前一律禁止

## 开发规范

- 提交前必须 `uv run pytest` 全绿，安全扫描不通过不算完成
- 费率假设变更必须同步更新 `research/costs.py` 与其测试
- 任何引入 `execution` 依赖的改动都会让 `test_safety.py` 失败
