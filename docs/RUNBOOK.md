# CoinTrader 操作手册

## 目录

1. [首次安装](#1-首次安装)
2. [日常使用流程](#2-日常使用流程)
3. [如何解读回测报告](#3-如何解读回测报告)
4. [安全操作规范](#4-安全操作规范)
5. [事故响应](#5-事故响应)
6. [常见问题](#6-常见问题)

---

## 1. 首次安装

### 1.1 环境

需要 `uv`。本项目用的是 uv 的**原生项目工作流**（`uv sync` / `uv run`），
不是 `uv pip install` 那一套。

```bash
cd CoinTrader
uv sync
```

`.python-version` 已固定为 `3.12`，`uv sync` 会自动安装该解释器（本机没有时），
并按 `uv.lock` 精确还原每个依赖的版本 —— 这就是为什么 `uv.lock`
**必须提交到版本库**：它是「三个月后还能复现同一个环境」的唯一保证。

### 1.2 自检

**不需要网络、不需要密钥**就能跑：

```bash
uv run cointrader doctor
```

预期输出（关键行）:

```
安全状态: 测试网 / 只读模式
[OK] 配置加载成功
[OK] 成本模型: 主流币往返 0.2900%
[OK] 执行滞后 1 期（无前瞻）
[OK] 风控限额: 单笔 200 / 单币 500 / 总 2000 USDT
自检通过。
```

任何 `[FAIL]` 都必须解决后才继续。

### 1.3 连通性

```bash
uv run cointrader ping
```

⚠️ **注意时钟偏移**。如果输出里出现

```
[警告] 时钟偏移 2730ms 超过 1 秒，真实交易会被拒绝。
```

说明本机时钟不准。这在**回测阶段不影响**（我们不签名请求），
但一旦要真实下单，币安会以 `-1021 Timestamp for this request is outside of the recvWindow` 拒绝。

修复（需要 root）:

```bash
sudo systemctl enable --now systemd-timesyncd
# 或
sudo chronyc makestep
```

验证:

```bash
timedatectl status | grep -i "synchronized\|NTP"
```

### 1.4 Python 环境自检

```bash
uv run pytest -q                  # 应全部通过
uv run ruff check src tests       # 应输出 All checks passed
uv run mypy src/cointrader        # 应输出 Success
```

改了依赖或 `pyproject.toml` 后，先 `uv lock` 更新锁文件、再 `uv sync` 落地到 `.venv`。
两步都不做，`uv run` 会悄悄用旧环境跑，排查起来最麻烦。

---

## 2. 日常使用流程

### 推荐的推进顺序

```
doctor  →  costs  →  scan  →  backtest  →  scenarios  →  （决策点）
自检       复核费率   选币      验证组合      评估仓位     是否继续
```

**不要跳步。** 尤其不要跳过 `costs` —— 费率假设错了，后面全部白做。

### 2.1 复核费率（每次改配置后必做）

```bash
uv run cointrader costs
```

**必须人工核对**：输出的费率是否与你账户的实际费率一致。
登录币安 → 费率页面 → 对比 `config/config.yaml` 的 `costs` 段。

差异示例：如果你有 VIP1 或更高的现货费率，必须更新配置，
否则回测会高估成本、低估收益（保守，但可能让你错过机会）。

### 2.2 扫描候选

```bash
# 全市场扫描（首次可能耗时数分钟，之后走缓存）
uv run cointrader scan --top 30

# 限制范围（快速试跑）
uv run cointrader scan --top 20 --max-symbols 100

# 导出 JSON 供进一步分析
uv run cointrader scan --top 50 --json reports/scan.json
```

**看哪个数字**：`年化(近)` 列。`年化(毛)` 是全历史平均，
会被历史行情污染，不能作为决策依据。策略组合回测使用固定滚动窗口；单币诊断也使用同一窗口。
组合回测还会在每个资金费事件重新排名，不能用全历史排名代替。

### 2.3 策略组合回测（自动选币）

```bash
uv run cointrader backtest --max-symbols 100 --json reports/portfolio.json
```

默认回测区间是最近 365 天（`backtest.history_days`）。自动发现可交易的 USDT 本位永续，
按可交易状态、排除币种和历史成交量过滤，再由回测引擎在每个资金费事件根据
最近 10 期平均资金费率排名。资金费和持仓统一按 8h 一期，最多同时持有
3 个币，每笔使用 33% 资金。基准情景每条现货/永续成交腿使用 0.15% 滑点，4 条腿往返滑点约 0.6%。

历史成交额来自已闭合 4h K 线最近 18 根（72 小时）的总 `quote_volume / 3`，即 3 天滑动平均日成交额；准入门槛为每天 100 万 USDT，不会用当前 ticker 回填历史。
当前资金费使用最近 10 期滑动平均值入场，入场至少 30% 年化；出场使用最近 30 期均值，均值转负退出。持仓 60 期内不因排名替换，61-120 期只有新币种均值高出当前 100% 才替换，超过 120 期需高出 70%。


报告目录还包含：

- `portfolio_ledger.csv`：组合逐 8h 收益、成本、敞口、净值
- `trade_ledger.csv`：所有交易逐 8h 资金费率、套利量、成本、累计收益
- `trade_*.csv`：每笔交易的逐期明细
- `portfolio_equity.svg`：组合总收益曲线
- `trade_*.svg`：每笔交易累计收益与对应 8h 资金费率双面板曲线

这些文件由回测 ledger 直接派生，图表不重复计算收益。

### 2.4 指定候选池的组合回测

```bash
uv run cointrader portfolio BTCUSDT ETHUSDT --json reports/portfolio.json
```

也可以不传币对，自动发现全市场候选。`portfolio BTCUSDT ETHUSDT` 只是限定候选池，
仍会按历史窗口动态选择，不是固定持有这两个币。
组合按资金费事件时间推进，在当前可见候选中排名；持仓和待成交订单都占用
`max_positions`。这和“先用全历史扫描结果选币，再把各币独立收益相加”不同，
后者会把未来赢家带回过去。组合要求 `max_positions × per_position_weight <= 1.0`。

### 2.5 破产情景分析

```bash
uv run cointrader scenarios --capital 5000 --exchange-exposure 1.0
```

如果你只打算用一家交易所，把 `--exchange-exposure` 设为 `1.0`，
你会看到"交易所暴雷 = 全部损失"，这通常是促使你分散的理由。

### 2.6 长期运行（服务器守护与断点重连）

`live run` 设计为服务器上长期不间断运行：主循环单轮异常只进 RECOVERY 不崩溃；
进程被 kill -9 / 关机 / 断电后重新拉起即可从上次状态继续（SQLite 事件账本 +
交易所对账恢复，历史数据零丢失）。

#### 2.6.1 systemd 守护部署与排障

```bash
# 安装（root；用户级 unit 加 --user）
deploy/install-systemd.sh install --project-dir /path/to/CoinTrader

# 查看状态 / 日志
deploy/install-systemd.sh status --project-dir /path/to/CoinTrader
journalctl -u cointrader -f

# 停止 / 重启（SIGTERM → 优雅停机：关流+释放锁+关闭 run_session）
systemctl stop cointrader
systemctl restart cointrader

# 卸载（回到 tmux 手工运行方式）
deploy/install-systemd.sh uninstall --project-dir /path/to/CoinTrader
```

排障要点：

- **崩溃循环**：连续 tick 异常达到 `execution.max_consecutive_tick_errors`（默认 10，
  5s 轮询 ≈ 50s 持续故障）后进程退出码 1，systemd `Restart=always` 拉起并重新
  预检/对账。若 600s 内重启超过 12 次（`StartLimitBurst=12`）systemd 停止拉起，
  `journalctl -u cointrader` 看最后错误，修复后 `systemctl reset-failed cointrader && systemctl start cointrader`。
- **看历史**：`cointrader live status` 的运行连续性块显示当前/历史 run、停止原因、
  未完结 pair 与锁持有者；`cointrader live pnl --all-runs` 看跨重启的 PnL 汇总。

#### 2.6.2 断点重连演练（kill -9 后重启）

```bash
# 1. 确认当前 run_id 与数据基线
cointrader live status
sqlite3 data/live/trading.sqlite3 "SELECT COUNT(*) FROM orders; SELECT COUNT(*) FROM fills;"

# 2. 非优雅杀掉进程（模拟 kill -9/断电）
pkill -9 -f 'cointrader.cli live run'
systemctl start cointrader   # 或重新执行 live run

# 3. 核验：旧会话被标 INTERRUPTED，新会话 RUNNING，数据行数不变
sqlite3 data/live/trading.sqlite3 \
  "SELECT run_id,status,stop_reason FROM run_sessions ORDER BY started_ms DESC LIMIT 2;"
cointrader live status       # 当前 run 已恢复；锁由新实例持有
```

预期：旧会话 `INTERRUPTED / process_exited_without_graceful_stop`；orders/fills 行数不变；
持仓从交易所对账恢复（日志「实际持仓」/「重启恢复」行）。

#### 2.6.3 demo ↔ 主网切换检查清单

模式切换四个要素，**缺一不可**（env 覆盖优先级高于 config，但不能绕过任何闸门）：

1. `execution.mode`（config）或 `COINTRADER_EXEC_MODE`（env，精确小写：paper/testnet/shadow/live）；
2. `COINTRADER_USE_TESTNET`（true=demo/测试网，false=主网）；
3. `COINTRADER_TRADING_ENABLED`（主网必须精确等于 `YES_I_AM_SURE`）；
4. `--confirm-live`（主网启动命令行显式确认）。

切换后必做：`cointrader live doctor` 确认「模式（来源）」与「端点」两行符合预期，
再 `systemctl restart cointrader`。密钥环境变量需对应切换（demo key / 主网 key）。

⚠️ **主网启动前必须走完开发设计文档 §13 完整清单**，本清单不替代 §13。

#### 2.6.4 KILL_SWITCH 在 systemd 下的路径说明

停机开关默认是**相对工作目录**的 `KILL_SWITCH` 文件。systemd 单元的
`WorkingDirectory` 固定为项目目录，因此 `touch <项目目录>/KILL_SWITCH` 即全局急停。
若想用绝对路径（任意目录都可触发），在 `.env` 里设 `COINTRADER_KILL_SWITCH_FILE` 为绝对路径。
急停后恢复仍需重新预检+对账（RECOVERY 不自动解除）。

---

## 3. 如何解读回测报告

### 3.1 先看两个收益率口径

报告里有两个数字，**不要混淆**：

| 口径 | 含义 | 什么时候看 |
|---|---|---|
| **占总资金年化** | 资金按策略实际使用率折算 | 评估这**笔投资**的整体回报 |
| **占投入资金年化** | 只算实际投出去的那部分钱 | 评估这**个策略**本身的优劣 |

例子：占资金 4.57%、占投入 32.70%、资金使用率 14.0%。

- 策略本身年化 32.7%，很不错
- 但只用了 14% 的资金，所以整体回报只有 4.57%
- **正确应对**：增加币种数量，提高资金利用率 —— 而不是放弃策略

### 3.2 结论判据（CLI 自动判定）

| 占投入年化 | 费用/毛收益 | 判定 |
|---|---|---|
| > 15% | < 50% | ✓ 值得进一步验证 |
| 5% ~ 15% | 任意 | ~ 边际，需要更好的条件 |
| 0 ~ 5% | 任意 | ✗ 收益被成本吃掉 |
| < 0 | 任意 | ✗ 策略不成立 |

### 3.3 必须检查的三个陷阱

**陷阱一：费用占毛收益比例过高**

如果输出显示 `费用/毛收益 > 50%`，说明你在给交易所打工。
这不是"收益低"，是"商业模式不成立"。

**陷阱二：收益高度集中**

```
⚠️ 收益高度集中：去掉少数几期就不赚钱，说明是运气而非策略
```

看到这行就要警惕。一个策略的全部收益来自 5 个结算期，那不是策略。

**陷阱三：样本内外差异巨大**

```
样本内 (350 期): 年化 3.45%
样本外 (150 期): 年化 7.23%
```

样本外**更好**是好事，但**样本外只允许看一次**。
如果你根据样本外表现回去调整参数，它立刻退化成样本内，
你的"验证"就变成了自我欺骗。

### 3.4 安慰剂检验怎么读

```
打乱后均值 : 1.0258%
p 值       : 0.9867
→ 真实顺序不优于随机排列（对资金费套利属正常）
```

**对资金费套利，这个结果是预期的、正常的。**

因为收益来自"持仓收租"，不来自"择时"。打乱顺序不影响总收租额。

真正需要警惕的是 **p 值很小**（真实收益远高于随机排列）——
那通常意味着收益集中在序列的特定位置，即来自某个事件，而非策略。

---

## 4. 安全操作规范

### 4.1 默认状态（无需任何操作）

程序默认**无法碰你的钱**：

- 数据层没有下单能力
- 执行层默认 `dry_run=True`
- 交易开关未设置 → 一切下单被拒绝

### 4.2 密钥管理（硬规则）

| 规则 | 说明 |
|---|---|
| **绝不**勾选"允许提现" | 创建 API Key 时核对 |
| **必须**绑定 IP 白名单 | 只放服务器出口 IP |
| 密钥只放环境变量 | 复制 `.env.example` 为 `.env`，填值 |
| `.env` 永不提交 | 已在 `.gitignore` |
| 定期轮换 | 泄露立刻在币安后台删除该 Key |

### 4.3 开启真实交易需要什么

**三道开关同时满足**，缺一不可：

```bash
# ① 精确的魔法字符串（'1'/'true'/'yes' 都不生效）
export COINTRADER_TRADING_ENABLED=YES_I_AM_SURE

# ② 明确关掉测试网
export COINTRADER_USE_TESTNET=false

# ③ 提供真实密钥
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
```

这个设计是刻意的：**它强迫你在下真单前明确地、有意识地做出决定**，
而不是顺手设个环境变量。

### 4.4 停机开关

任何时候想让程序立刻停止下单：

```bash
touch KILL_SWITCH
```

这是**文件系统级**的检查，不需要改代码、不需要重启进程，立即生效。

恢复：

```bash
rm KILL_SWITCH
```

**建议**：在真实交易时，把停机命令写好贴在终端里，需要时直接回车。

### 4.5 审计日志

所有下单尝试（**包括被拒绝的**）都写入 `logs/audit.log`：

```bash
# 查看今天被拒绝的下单
grep '"authorized":false' logs/audit.log | tail -20

# 查看所有真实下单授权（非干跑）
grep '"authorized":true' logs/audit.log | grep '"dry_run":false'
```

**日志中永不含密钥明文**，只有指纹（`fp:xxxxxxxxxxxx`），
可用于对账但无法反推密钥。

### 4.6 变更前的检查清单

修改 `config/config.yaml` 后：

- [ ] `doctor` 通过
- [ ] `costs` 输出的费率与实际账户一致
- [ ] 如果改了 `backtest.execution_lag_bars`，确认它 **>= 1**
- [ ] 如果改了 `backtest.rolling_window_periods`，确认它为正且不小于 `strategy.entry.lookback_periods`
- [ ] 组合回测确认 `max_positions × per_position_weight <= 1.0`，否则回测会拒绝超配
- [ ] 如果改了 `risk` 段的限额，确认层级正确（单笔 ≤ 单币种 ≤ 总敞口）
- [ ] `pytest` 全绿

---

## 5. 事故响应

### 5.1 发现持仓未对冲（最紧急）

**症状**：日志出现 `⚠️ 裸头寸！` 或风控告警 `未完全对冲`

**立即行动**：

1. 立刻 `touch KILL_SWITCH` 停止所有自动下单
2. 登录交易所**手动**查看实际持仓
3. 判断哪一腿缺失：
   - 只有现货 → 手动卖出对应数量的现货，或手动开永续空头补上
   - 只有永续 → 手动买入现货补上，或平掉永续
4. **不要**急着让程序自动补 —— 先搞清楚为什么失败
5. 记录事故：时间、币种、方向、失败原因、处理方式

**为什么紧急**：裸头寸是资金费套利唯一会真正亏大钱的方式。
一次没对冲上的暴跌能抹掉几个月的收益。

### 5.2 收到 HTTP 418（IP 被封）

**症状**：`IPBanError: IP 已被币安临时封禁`

**行动**：

1. **立即停止所有请求**。继续请求会延长封禁
2. 检查是否有多进程/多机器共用同一出口 IP
3. 等待封禁解除（通常几分钟到几小时）
4. 恢复后**降低请求频率**：
   - 调低 `config.yaml` 的 `data.rate_limit.soft_limit_ratio`（如 0.80 → 0.60）
   - 检查是否有脚本绕过了缓存

### 5.3 触及日亏损限额

**症状**：`24h 亏损 X% 已达停机线`

**行动**：

1. 程序已自动停止开新仓，检查是否已平仓
2. **不要**立即重启并放宽限额 —— 先搞清楚为什么亏
3. 复盘：是对冲失效？基差突变？还是参数问题？
4. 修改配置后，`touch KILL_SWITCH` 保持停机状态，
   直到你理解并解决了根因

### 5.4 交易所异常（停提币、公告异常）

**行动**：

1. `touch KILL_SWITCH`
2. 关注官方公告渠道，**不要**依据小道消息操作
3. 如果只是提币暂停但交易正常，通常不需要平仓
   （平仓会实现损失，而暂停往往是暂时的）
4. 如果涉及偿付能力问题，参考 `scenarios` 的输出评估损失

### 5.5 数据异常

**症状**：回测结果与之前差异巨大

**行动**：

1. 检查缓存是否损坏：`ls -la data/cache/`
2. 清理缓存重跑：`rm -rf data/cache/funding_history/`
3. 对比币安网页端的资金费数据，确认是数据源问题还是代码问题
4. 币安**会修订历史资金费**，所以小幅差异是正常的

---

## 6. 常见问题

### Q: 为什么 BTCUSDT 回测显示零交易？

因为 BTC 的资金费年化通常只有 10% 左右，低于配置里的进场门槛
（`strategy.entry.min_trailing_annualized: 0.20`）。

**这是正确行为** —— 策略在说实话："这个币不值得做"。

想看它交易，可以放宽门槛，但要想清楚为什么：

```yaml
strategy:
  entry:
    min_trailing_annualized: 0.05   # 从 0.20 降到 0.05
```

降低门槛意味着接受更低的收益，需要确认扣费后仍为正。

### Q: 扫描出来一堆年化 100%+ 的币，为什么回测只有几个点？

因为**年化(毛) 是全历史平均，会被极值拉高**。

例子：AINUSDT 的均值年化 34.55%，但中位数只有 15.0% ——
少数几期的极端高费率把平均值拉起来了。而策略只在满足条件时才持仓，
不会全程享受那个平均值。

**看 `年化(近)` 和中位数，别只看均值。**

### Q: 4h 和 8h 结算有什么区别？

4h 结算 = 一年收 2190 次租，8h = 1095 次。

同样的单期费率下，4h 的年化是 8h 的**两倍**。

所以扫描器必须逐币使用各自的周期。这也是
`data/funding.py` 的 `annualize_rate()` 强制要求传 `interval_hours` 的原因。

### Q: 为什么回测赚钱但我不确定能不能实盘？

因为回测**没有**建模这些东西：

- 极端行情下的滑点（回测用的是历史平均，可能差 10 倍）
- 订单部分成交（回测假设全部成交）
- 交易所故障时的无法下单
- 你的实际费率可能比配置的高

**正确做法**：先用 `backtest/scenarios.py` 的悲观情景跑一遍。
如果只有乐观成本下才赚钱，那就是在赚假设的钱。

### Q: 时钟偏移警告要紧吗？

**回测阶段**：不要紧，我们不签请求。

**真实交易阶段**：非常要紧。币安会拒绝时间戳偏差超过
`recvWindow`（默认 5000ms）的请求，错误码 `-1021`。
表现是"所有下单都失败"，很难一眼看出原因。

修好系统时间同步即可，见 §1.3。

### Q: 我可以直接改 config.yaml 里的费率吗？

可以，**而且应该** —— 如果你在币安的实际费率与默认值不同。

但改完必须：

1. 跑 `pytest tests/test_costs.py` 确认成本模型没被改坏
2. 跑 `costs` 命令人工复核输出
3. 在回测报告里记录用的是哪套费率假设

费率假设是回测结论的前提。前提变了，结论就变了。

### Q: 这个框架能自动交易吗？

**当前不能，这是刻意的。**

执行层的代码存在（`execution/`），但：

- 没有实现真实的交易所签名
- `Exchange` 只有 `SimulatedExchange` 一个实现
- CLI 里没有 `trade` 子命令
- 所有下单默认被闸门拒绝

真正上线需要先完成 **M5 决策点**：回测证明扣费后收益值得投入。
在那之前，任何"让它自动跑起来"的尝试都是在没有证据的情况下冒险。

### Q: 怎么知道我的改动没有破坏安全约束？

```bash
uv run pytest tests/test_safety.py -v
```

45 条安全测试覆盖静态扫描与运行时行为。
**任何一条红了都不算完成。**

---

## 附：关键文件速查

| 想知道什么 | 看哪里 |
|---|---|
| 整体架构与安全模型 | `docs/ARCHITECTURE.md` |
| 成本怎么算的 | `src/cointrader/research/costs.py` |
| 年化公式 | `src/cointrader/data/funding.py` 的 `annualize_rate` |
| 回测的无前瞻保证 | `src/cointrader/backtest/engine.py` 模块文档 |
| 安全闸门逻辑 | `src/cointrader/execution/guard.py` |
| 破产情景与仓位建议 | `src/cointrader/backtest/scenarios.py` |
| 所有可调参数 | `config/config.yaml` |
