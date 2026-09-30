"""交易场地（venue）—— 币安主网与 Demo Trading 的币池是两套独立清单。

## 为什么必须区分 venue

2026-09 实测两个端点各自的 ``exchangeInfo``：

    主网 fapi.binance.com      527 个可交易 USDT 永续
    Demo demo-fapi.binance.com 528 个可交易 USDT 永续
    仅主网有 65 个（PUMPUSDT / XAUTUSDT / ICPUSDT / MANTRAUSDT ...）
    仅 Demo 有 66 个（TONUSDT / IPUSDT / LRCUSDT / ICXUSDT / SYSUSDT ...）

Demo 不是主网的镜像 —— 它是**独立的合约清单**。把主网 ``exchangeInfo`` 选出的
symbol 拿去 Demo 下单会直接失败（"合约不存在"），反之同理。所以 venue 必须
显式携带，且**缓存键与历史覆盖 namespace 都要带 venue**，否则两个场地互相
污染：Demo 拉的 K 线/资金费会被主网请求当成已覆盖而跳过。

## 回测契约（不可违反）

**回测永远使用主网历史。** 本项目的历史数据源（K 线、资金费、成交额）只有
主网一份；Demo 的历史不参与任何回测。因此：

- ``backtest/`` 下的代码不接收 ``Venue`` 参数，一律走 ``Venue.MAINNET``；
- ``data/coverage.py`` 的历史覆盖索引用 ``mainnet`` 作 venue，
  绝不允许把 Demo 池喂给回测。

这条契约的动机是数据真实性，不是便利性：Demo 的退市合约仍返回行情，
用它的池子做回测会得到主网上无法复现的结论。
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "MAINNET_VENUE",
    "SUPPORTED_VENUES",
    "Venue",
    "venue_for_execution_mode",
]


class Venue(str, Enum):
    """币安数据/交易场地。

    值同时用作缓存 namespace 后缀与历史覆盖的 ``CoverageKey.venue``，
    因此**不可随意重命名**（改名 = 旧缓存整体失效，按 miss 重拉）。
    """

    MAINNET = "mainnet"
    """币安主网：``api.binance.com`` / ``fapi.binance.com``。"""

    DEMO = "demo"
    """币安 Demo Trading：``demo-api.binance.com`` / ``demo-fapi.binance.com``。

    官方模拟盘，一对 key 通用现货+合约，虚拟资金；行情与主网接近但
    **合约清单不同**，且用户数据流不可用（必须走 REST 轮询）。
    """


#: 历史数据恒定的 venue（回测唯一允许的场地，见模块 docstring「回测契约」）。
MAINNET_VENUE: Venue = Venue.MAINNET

#: 全部受支持的 venue，供配置校验使用。
SUPPORTED_VENUES: tuple[Venue, ...] = (Venue.MAINNET, Venue.DEMO)

#: ``execution.mode`` → venue。paper/testnet/shadow 都连 testnet 域名，
#: 而 ``api.spot_testnet_base`` / ``api.futures_testnet_base`` 当前指向
#: Demo Trading 域名（见 config.yaml api 节注释），故三者统一归 DEMO；
#: 只有 live 触达主网。
_MODE_TO_VENUE: dict[str, Venue] = {
    "paper": Venue.DEMO,
    "testnet": Venue.DEMO,
    "shadow": Venue.DEMO,
    "live": Venue.MAINNET,
}


def venue_for_execution_mode(mode: str) -> Venue:
    """由 ``execution.mode`` 推导 venue。

    Args:
        mode: ``paper`` / ``testnet`` / ``shadow`` / ``live``。

    Returns:
        对应 venue。

    Raises:
        ValueError: 未知 mode（调用方应先经 ``ExecutionConfig`` 校验，
            这里是纵深防御：宁可报错也不要静默退回主网）。
    """
    try:
        return _MODE_TO_VENUE[mode.lower()]
    except KeyError as exc:
        raise ValueError(
            f"未知 execution.mode: {mode!r}（可选 {sorted(_MODE_TO_VENUE)}）"
        ) from exc
