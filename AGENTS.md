# 仓库指南

币安资金费率套利（funding rate carry）研究框架，Python 3.12 + uv，入口 `cointrader`（`src/cointrader/cli.py`）。

## 结构

- `config/config.yaml`：唯一配置源，不含密钥；密钥只从环境变量读（模板 `.env.example`），永不落盘。
- `src/cointrader/`：`data/`（只读公开 API）、`research/`（成本/指标）、`backtest/`（回测+情景）、`execution/`（默认禁用的执行层）、`live/`（实盘编排）、`webui/`、`reporting/`。
- 依赖方向严格单向：`data/research/backtest` 禁止 import `execution`，只有 `live` 可同时调用三者。

## 命令

```bash
uv sync                      # 装依赖（dev 组默认包含）
uv run pytest                # 默认跳过联网测试；联网测试用 -m network
uv run ruff check .          # lint
uv run mypy                  # 类型检查
uv run cointrader doctor     # 自检（离线）
uv run cointrader backtest   # 组合回测，报告默认写 reports/backtest/
```

## 硬性规则

- 提交前 `uv run pytest` 必须全绿；`tests/test_safety.py`（安全静态扫描）与 `tests/test_no_lookahead.py`（前瞻偏差证伪）失败即未完成。
- 不要引入让下层包依赖 `execution` 的改动；会破坏依赖方向约束。
- 费率假设变更必须同步改 `research/costs.py` 及其测试。
- 真实下单有三重闸门：`COINTRADER_TRADING_ENABLED=YES_I_AM_SURE`、`COINTRADER_USE_TESTNET=false`、`--confirm-live`；`touch KILL_SWITCH` 全局停机。不要绕过或弱化任何闸门。
- 内部包统一用相对 import（ruff 忽略 TID252 是刻意设计）；datetime 必须带时区（DTZ）。
- 回测严禁使用 `t` 时刻之后的数据（无前瞻偏差）；选币只用截至 `t` 已闭合的 K 线。

## 部署流程（更新项目一律照此操作）

用户要求更新本项目时，固定顺序：

```text
1. commit      在 develop 分支提交（uv run pytest 全绿后）
2. merge       git checkout main && git merge develop（通常 fast-forward）
3. push        git push origin main develop
4. VPS pull    见下，拉取后重启服务并验证
```

VPS 部署细节：

- 设备 `vps-ali-lrt`（100.100.1.21，Tailscale），代码目录 `/home/vps-ali-lrt/CoinTrader`，守护 `cointrader.service`（systemd，`Restart=always`，以 `vps-ali-lrt` 用户运行）。
- VPS 上 `git pull`（main 分支）；若依赖有变补 `uv sync`。
- 重启：`systemctl restart cointrader`（需 root，用 `vps-ali-root` 账户）。
- 验证：`journalctl -u cointrader -n 20` + `uv run cointrader live status`（服务 RUNNING、对账一致）；WebUI `http://100.100.1.21:8888/`（Tailscale）。
- VPS 的 `.env` 单独维护（含密钥与 `HTTPS_PROXY=http://127.0.0.1:7897`），不进 git、不被 pull 覆盖。
- 急停：VPS 上 `touch /home/vps-ali-lrt/CoinTrader/KILL_SWITCH`（全局拒单，无需重启）。
