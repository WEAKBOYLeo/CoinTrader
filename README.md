# CoinTrader

币安加密货币自动化交易研究框架。当前聚焦一个策略：**资金费率套利（funding rate carry）**——现货买入 + 永续卖出的对冲套利，赚取资金费。

## 这个项目要回答的问题

在扣除真实成本之后，资金费套利还有净收益吗？

在整个研究框架跑完之前，不写任何执行层代码。框架存在的意义是**证伪**，不是确认。
如果回测显示扣费后收益微薄，正确的结论是「不值得做」。

## 快速开始

```bash
# 1. 环境（需要 uv；按 .python-version 装 3.12，按 uv.lock 精确还原依赖，默认含 dev 工具链）
uv sync

# 2. 自检（不需要密钥、不需要联网）
uv run cointrader doctor

# 3. 跑测试（默认不跑联网测试；联网测试用 -m network 显式启用）
uv run pytest

# 4. 探活币安接口（需要联网，检查连通性与时钟偏移）
uv run cointrader ping

# 5. 成本模型明细
uv run cointrader costs

# 6. 扫描全市场资金费候选（实时）
uv run cointrader scan

# 7. 科学策略组合回测：自动发现 USDT 永续、成交额过滤、动态选币
uv run cointrader backtest --max-symbols 100 --json reports/portfolio.json
# 报告默认写入 reports/backtest/；可指定目录：
uv run cointrader backtest --report-dir reports/backtest-365

# 8. 指定候选池的组合诊断（仍然动态选币，不是固定持有）
uv run cointrader portfolio BTCUSDT ETHUSDT

# 9. 破产情景分析与仓位建议
uv run cointrader scenarios
```

## 回测模型

完整参数以 `config/config.yaml` 为准，要点：

- **预热**：默认 `backtest.rolling_window_periods: 30` 固定历史预热，窗口未满不交易；
  默认数据范围是最近 365 天（`backtest.history_days`），并非无限历史。
- **信号**：入场看最近 10 期资金费率滑动平均（年化 ≥ 30%），出场看最近 30 期均值转负；
  替换按持仓年龄分段设门槛，并禁止过早替换。
- **延迟**：信号产生后至少延迟 `execution_lag_bars` 期执行，执行前信号失效会撤单。
- **时间轴**：不同结算周期（1h/4h/8h）的资金费事件先聚合到统一 8h 时间轴；
  每个 8h 桶只含该桶结束时已发生的事件，缺失桶补 0；
  所有交易次数、持仓期数和年化都按 8h（每年 1095 期）计算。
- **选币**：自动发现 USDT 本位永续，用历史 4h K 线最近 18 根（72 小时）滚动平均日成交额过滤；
  时点 `t` 只使用截至 `t` 已闭合的 K 线，不用当前 24h 成交额回填历史；
  最多持有 `max_positions` 个币（默认 3），每笔 `per_position_weight`（默认 33%）。
- **因果边界**：资金费滚动均值、连续正负费率、排名只读取事件当时及之前的数据。
  `tests/test_no_lookahead.py` 静态证伪前瞻偏差。
- **成本**：基准情景每条现货/永续成交腿 0.15% 滑点；一笔对冲往返含 4 条腿，
  滑点约名义额 0.6%，另加手续费（`costs` 命令可打印明细）。
  默认回测还把 `risk.max_basis_adverse_pct` 应用于每笔交易的进出场基差损失。

## 安全模型

**默认状态下这个程序没有能力碰你的钱。**

三层隔离：

1. **类型隔离** —— `data/` 层的客户端没有任何下单能力，也没有 API 凭证参数，
   它甚至无法签名请求。
2. **运行时闸门** —— 真实下单需要环境变量精确等于 `YES_I_AM_SURE`
   （写 `1` / `true` / `yes` 都不生效，这是刻意的摩擦），
   且默认走测试网，且 `KILL_SWITCH` 文件不存在（`touch KILL_SWITCH` 即全局停机）。
3. **默认干跑** —— 执行层所有方法默认 `dry_run=True`。

静态约束由 `tests/test_safety.py` 强制，测试失败即构建失败。

**密钥永不落盘。** 只从环境变量读取（模板见 `.env.example`），
且 `SecretStr` 包装保证 `repr()` / `str()` / f-string 都只输出指纹，不输出明文。

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
  reporting/   报告导出
tests/         含安全静态扫描与前瞻偏差证伪
```

依赖方向严格单向：`data/`、`research/`、`backtest/` 不得 import `execution/`；
只有 `live/` 可以同时调用 `data/`、`research/` 与 `execution/`（`test_safety.py` 静态验证）。

## 实盘执行（分阶段，见 docs/开发设计文档.md）

执行层已实现 M1-M3（签名 REST/限流、Decimal 规则归一化、SQLite 事件账本、
订单状态机、双腿开平仓与 UNKNOWN 恢复、用户数据流、对账器），并带完整单测。
运行模式在 `config.yaml → execution.mode`：

| 模式 | 行为 |
|---|---|
| `paper` | 不发私有请求；模拟走 backtest/broker 路径 |
| `testnet`（当前默认） | 测试网真实订单。当前端点指向币安 **Demo Trading**（demo.binance.com，官方模拟盘，一对 key 通用现货+合约）；因 demo 用户流不可用，用 `user_stream_mode: poll` 轮询 |
| `shadow` | 读真实行情/账户，只生成意图不下单 |
| `live` | 主网真实订单：需 `COINTRADER_TRADING_ENABLED=YES_I_AM_SURE` + `COINTRADER_USE_TESTNET=false` + `--confirm-live` 三重确认 |

env 覆盖：`COINTRADER_EXEC_MODE`（精确小写 `paper/testnet/shadow/live`）可单点覆盖上表模式，
`live doctor` / `live status` 会显示「模式来源（config / env）」与所选端点；
env 不能绕过任何闸门（live 仍需上述三重确认）。

常用命令：

```bash
uv run cointrader live doctor    # 启动预检：时钟偏移/凭证/停机开关/模式+端点
uv run cointrader live run       # 主循环（Ctrl-C/SIGTERM 优雅停机：关流+释放单实例锁+关闭会话）
uv run cointrader live status    # 只读：服务/账户/对账/流新鲜度/风险状态 + 运行连续性块
uv run cointrader live positions # 只读：当前策略持仓与未实现 PnL
uv run cointrader live orders    # 只读：订单历史与状态
uv run cointrader live trades    # 只读：pair 交易/两腿成交与手续费
uv run cointrader live pnl       # 只读：PnL 分项；--all-runs 跨所有 run 聚合（含跨 run 未平仓）
uv run cointrader live report    # 只读：导出完整报告数据包
```

### 长期运行（服务器守护）

`live run` 支持服务器长期不间断运行：主循环单轮异常只进 RECOVERY 不崩溃；
进程被 kill -9 / 关机 / 断电后重新拉起即从上次状态继续（SQLite 事件账本 + 交易所对账恢复，
旧会话自动标 `INTERRUPTED`，历史数据零丢失）。

```bash
# systemd 守护（root；用户级 unit 加 --user）
deploy/install-systemd.sh install --project-dir /path/to/CoinTrader
deploy/install-systemd.sh status --project-dir /path/to/CoinTrader   # systemctl status + 最近日志
systemctl stop cointrader   # SIGTERM → 优雅停机

# 崩溃/断电后：systemd 自动拉起（Restart=always）；手动也可 systemctl start cointrader
# 跨重启查看状态与 PnL
uv run cointrader live status          # 当前/历史 run、未完结 pair、锁持有者
uv run cointrader live pnl --all-runs  # 跨所有 run 的 PnL 汇总

# 急停：touch KILL_SWITCH（systemd 下 = 项目目录/KILL_SWITCH）
```

排障与断点重连演练步骤见 docs/RUNBOOK.md §2.6。

⚠️ 当前处于 **M4（Demo Trading 端到端 7 天长跑）** 阶段（见 docs/开发日志.md）。
M4 与 M6（主网 canary）尚未通过，`live run` 目前只应在 demo 环境使用；
主网启动必须走完开发设计文档 §13 清单，在此之前主网放量（M7）一律禁止。

## 已知限制

- 币安只保留约 **1 年**资金费历史，回测跨度受此限制
- 资金费结算周期有 1h / 4h / 8h 三种，年化必须按各合约自己的周期折算
- 回测用的是历史平均滑点，极端行情下可能差 10 倍
- 交易所尾部风险无法对冲，只能靠仓位控制（见 `cointrader scenarios`）
- Demo Trading 用户数据流不可用，poll 模式下成交/撤单感知延迟高于 WS 模式；
  M6 主网 canary 必须回到 stream 模式
- 实盘链路尚未经长跑验证（M4 进行中）

## 开发规范

- 提交前必须 `uv run pytest` 全绿，安全扫描不通过不算完成
- 费率假设变更必须同步更新 `research/costs.py` 与其测试
- 任何引入 `execution` 依赖到下层包的改动都会让 `test_safety.py` 失败
- 配置改动只改 `config/config.yaml`；密钥永远不进配置文件或代码
