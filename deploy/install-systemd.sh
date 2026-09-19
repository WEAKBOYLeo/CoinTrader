#!/usr/bin/env bash
# CoinTrader systemd 守护安装脚本（计划 1.0 T3 / AC-05）。
#
# 子命令：
#   install   校验 → 渲染 unit → 写入 → daemon-reload → enable --now（幂等）
#   uninstall disable --now → 删除 unit → daemon-reload
#   verify    离线校验（bash 语法 / .venv 可用 / 渲染无占位符残留 / 可选 systemd-analyze）
#             —— 失败退出非 0，且不写 /etc
#   status    systemctl status + 最近日志
#
# 选项：
#   --project-dir <dir>   项目目录（install/uninstall/verify 必填）
#   --user                安装用户级 unit（~/.config/systemd/user），否则系统级（需 root）
set -euo pipefail

UNIT_NAME="cointrader"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/cointrader.service"

PROJECT_DIR=""
USER_MODE=0
SUBCOMMAND=""

fail() { echo "  ❌ $*" >&2; exit 1; }
pass() { echo "  ✅ $*"; }
info() { echo "  ℹ️  $*"; }

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

# ---------------------------------------------------------------- 参数解析
[ $# -ge 1 ] || usage
SUBCOMMAND="$1"
shift
case "$SUBCOMMAND" in
  install|uninstall|verify|status) ;;
  -h|--help|help) usage ;;
  *) usage ;;
esac

while [ $# -gt 0 ]; do
  case "$1" in
    --project-dir)
      [ $# -ge 2 ] || { echo "--project-dir 缺少参数" >&2; exit 2; }
      PROJECT_DIR="$2"
      shift 2
      ;;
    --user)
      USER_MODE=1
      shift
      ;;
    *)
      echo "未知参数: $1" >&2
      usage
      ;;
  esac
done

if [ "$SUBCOMMAND" != "status" ] && [ -z "$PROJECT_DIR" ]; then
  fail "${SUBCOMMAND} 需要 --project-dir <dir>"
fi

# ---------------------------------------------------------------- 工具函数
ctl() {
  if [ "$USER_MODE" -eq 1 ]; then
    systemctl --user "$@"
  else
    systemctl "$@"
  fi
}

journal() {
  if [ "$USER_MODE" -eq 1 ]; then
    journalctl --user -u "$UNIT_NAME" "$@"
  else
    journalctl -u "$UNIT_NAME" "$@"
  fi
}

unit_file() {
  if [ "$USER_MODE" -eq 1 ]; then
    echo "${HOME}/.config/systemd/user/${UNIT_NAME}.service"
  else
    echo "/etc/systemd/system/${UNIT_NAME}.service"
  fi
}

require_root_or_user() {
  if [ "$USER_MODE" -eq 0 ] && [ "$(id -u)" -ne 0 ]; then
    fail "系统级 unit 需要 root（或改用 --user 安装用户级 unit）"
  fi
}

render_to() {
  local dest="$1"
  sed "s|@PROJECT_DIR@|${PROJECT_DIR}|g" "$TEMPLATE" > "$dest"
}

# ---------------------------------------------------------------- verify
cmd_verify() {
  echo "  【verify】离线校验（不写 /etc）"
  # 1) 脚本自身语法
  if bash -n "$0"; then
    pass "脚本语法 (bash -n)"
  else
    fail "脚本语法错误"
  fi
  # 2) 模板存在且含占位符
  [ -f "$TEMPLATE" ] || fail "单元模板不存在: $TEMPLATE"
  grep -q "@PROJECT_DIR@" "$TEMPLATE" || fail "模板缺少 @PROJECT_DIR@ 占位符"
  pass "单元模板存在且含占位符"
  # 3) 项目目录
  [ -d "$PROJECT_DIR" ] || fail "项目目录不存在: $PROJECT_DIR"
  pass "项目目录存在: $PROJECT_DIR"
  # 4) .venv 可用
  local vpy="${PROJECT_DIR}/.venv/bin/python"
  [ -x "$vpy" ] || fail ".venv/bin/python 不存在或不可执行（先 uv sync）: $vpy"
  "$vpy" --version >/dev/null 2>&1 || fail ".venv/bin/python 无法运行 --version"
  pass ".venv Python: $("$(dirname "$vpy")/python" --version 2>&1)"
  # 5) CLI 可跑
  if (cd "$PROJECT_DIR" && ./.venv/bin/python -m cointrader.cli --version) >/dev/null 2>&1; then
    pass "cointrader --version 可运行"
  else
    fail "cointrader --version 运行失败（uv sync 后重试）"
  fi
  # 6) 渲染到临时文件，校验无占位符残留
  local rendered
  rendered="$(mktemp "${TMPDIR:-/tmp}/cointrader-XXXXXX.service")"  # systemd-analyze 要求文件名以 .service 结尾
  trap 'rm -f "$rendered"' RETURN
  render_to "$rendered"
  if grep -q "@PROJECT_DIR@" "$rendered"; then
    fail "渲染后仍有 @PROJECT_DIR@ 残留（PROJECT_DIR 含分隔符？）"
  fi
  pass "渲染无占位符残留"
  # 7) systemd-analyze（可选）
  if command -v systemd-analyze >/dev/null 2>&1; then
    if systemd-analyze verify "$rendered" >/dev/null 2>&1; then
      pass "systemd-analyze verify"
    else
      info "systemd-analyze verify 未通过或无权限（可能非 systemd 环境/非 root），跳过"
    fi
  else
    info "systemd-analyze 不可用，跳过"
  fi
  echo "  ✅ verify 通过"
}

# ---------------------------------------------------------------- install
cmd_install() {
  require_root_or_user
  cmd_verify
  local dest
  dest="$(unit_file)"
  # 幂等：重复安装先停旧 unit
  if ctl is-active --quiet "$UNIT_NAME" 2>/dev/null; then
    info "已有运行中的 ${UNIT_NAME}，先停止"
    ctl stop "$UNIT_NAME"
  fi
  mkdir -p "$(dirname "$dest")"
  render_to "$dest"
  grep -q "@PROJECT_DIR@" "$dest" && fail "安装产物仍有 @PROJECT_DIR@ 残留，已中止"
  if [ "$USER_MODE" -eq 1 ]; then
    mkdir -p "${HOME}/.config/systemd/user"
  fi
  ctl daemon-reload
  ctl enable --now "$UNIT_NAME"
  pass "已安装并启动: $dest"
  info "查看状态: $0 status $( [ -n "$PROJECT_DIR" ] && echo "--project-dir $PROJECT_DIR" ) $( [ "$USER_MODE" -eq 1 ] && echo "--user" )"
}

# ---------------------------------------------------------------- uninstall
cmd_uninstall() {
  require_root_or_user
  local dest
  dest="$(unit_file)"
  ctl disable --now "$UNIT_NAME" 2>/dev/null || info "unit 未在运行（跳过 disable --now）"
  if [ -f "$dest" ]; then
    rm -f "$dest"
    pass "已删除 unit: $dest"
  else
    info "unit 文件不存在: $dest"
  fi
  ctl daemon-reload
  pass "uninstall 完成（回到 tmux 手工运行方式）"
}

# ---------------------------------------------------------------- status
cmd_status() {
  ctl status "$UNIT_NAME" --no-pager || info "unit 未激活（is-active 检查失败时常见）"
  echo "  ---- 最近 20 条日志 ----"
  journal -n 20 --no-pager || info "journalctl 无日志"
}

case "$SUBCOMMAND" in
  verify)    cmd_verify ;;
  install)   cmd_install ;;
  uninstall) cmd_uninstall ;;
  status)    cmd_status ;;
esac
