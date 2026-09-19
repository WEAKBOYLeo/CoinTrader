"""实盘编排层（live）。

上层编排：可同时调用 ``data/``（只读行情）、研究策略与 ``execution/``（执行）。
``data/``、``research/``、``backtest/`` 继续禁止反向 import ``execution/``。
"""

from .portfolio import Signal, build_signal
from .service import LiveService, ServiceState, StartupReport

__all__ = ["LiveService", "ServiceState", "Signal", "StartupReport", "build_signal"]
