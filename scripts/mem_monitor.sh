#!/bin/bash
# Memory Monitor - 内存监控脚本
# 功能：定期检查内存使用率，超过阈值时告警或自动kill最大内存进程
# 用法：nohup bash scripts/mem_monitor.sh &

# ===== 配置 =====
WARN_THRESHOLD=85      # 告警阈值(%)：超过此值发出警告
KILL_THRESHOLD=93      # Kill阈值(%)：超过此值自动kill最大用户进程
CHECK_INTERVAL=30      # 检查间隔(秒)
LOG_FILE="$HOME/mem_monitor.log"
# 保护进程列表（这些进程不会被kill）
PROTECTED="sshd|bash|zsh|systemd|mem_monitor|claude"

# ===== 函数 =====
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

get_mem_usage_pct() {
    free | awk '/^Mem:/ {printf "%.1f", $3/$2 * 100}'
}

get_swap_usage_pct() {
    free | awk '/^Swap:/ {if($2==0) print "0.0"; else printf "%.1f", $3/$2 * 100}'
}

get_top_mem_process() {
    # 获取当前用户占用内存最多的进程（排除保护列表）
    ps -u "$USER" -o pid,rss,%mem,comm --sort=-rss --no-headers | \
        grep -vE "$PROTECTED" | head -1
}

get_top_n_processes() {
    ps -u "$USER" -o pid,rss,%mem,comm --sort=-rss --no-headers | head -"${1:-5}"
}

bytes_to_human() {
    local kb=$1
    if [ "$kb" -ge 1048576 ]; then
        echo "$(awk "BEGIN{printf \"%.1f\", $kb/1048576}")GB"
    elif [ "$kb" -ge 1024 ]; then
        echo "$(awk "BEGIN{printf \"%.1f\", $kb/1024}")MB"
    else
        echo "${kb}KB"
    fi
}

# ===== 主循环 =====
log_msg "=========================================="
log_msg "Memory Monitor 启动"
log_msg "告警阈值: ${WARN_THRESHOLD}% | Kill阈值: ${KILL_THRESHOLD}%"
log_msg "检查间隔: ${CHECK_INTERVAL}秒"
log_msg "=========================================="

WARN_SENT=0
LAST_WARN_TIME=0

while true; do
    MEM_PCT=$(get_mem_usage_pct)
    SWAP_PCT=$(get_swap_usage_pct)
    MEM_INT=${MEM_PCT%.*}

    if [ "$MEM_INT" -ge "$KILL_THRESHOLD" ]; then
        # 超过 kill 阈值 - 自动 kill 最大进程
        TOP_PROC=$(get_top_mem_process)
        if [ -n "$TOP_PROC" ]; then
            TOP_PID=$(echo "$TOP_PROC" | awk '{print $1}')
            TOP_RSS=$(echo "$TOP_PROC" | awk '{print $2}')
            TOP_NAME=$(echo "$TOP_PROC" | awk '{print $4}')
            TOP_RSS_H=$(bytes_to_human "$TOP_RSS")

            log_msg "⚠️  危险! 内存使用 ${MEM_PCT}% (Swap: ${SWAP_PCT}%)"
            log_msg "🔪 正在 Kill 最大进程: PID=$TOP_PID 名称=$TOP_NAME 内存=$TOP_RSS_H"

            kill -15 "$TOP_PID" 2>/dev/null
            sleep 3
            # 如果还没死，强制kill
            if kill -0 "$TOP_PID" 2>/dev/null; then
                kill -9 "$TOP_PID" 2>/dev/null
                log_msg "🔪 强制Kill: PID=$TOP_PID"
            fi

            log_msg "Kill完成，当前内存占用top5:"
            get_top_n_processes 5 | while read -r line; do
                log_msg "  $line"
            done
        fi
        WARN_SENT=0

    elif [ "$MEM_INT" -ge "$WARN_THRESHOLD" ]; then
        # 超过告警阈值 - 发出警告（每5分钟最多一次）
        NOW=$(date +%s)
        if [ $((NOW - LAST_WARN_TIME)) -ge 300 ] || [ "$WARN_SENT" -eq 0 ]; then
            log_msg "⚠️  警告! 内存使用 ${MEM_PCT}% (Swap: ${SWAP_PCT}%)"
            log_msg "当前内存占用top5:"
            get_top_n_processes 5 | while read -r line; do
                log_msg "  $line"
            done

            # 终端弹窗提醒（如果有 wall 权限）
            echo "⚠️ 内存告警: ${MEM_PCT}% 已使用! 请检查进程。" | wall 2>/dev/null

            WARN_SENT=1
            LAST_WARN_TIME=$NOW
        fi

    else
        # 内存正常
        if [ "$WARN_SENT" -eq 1 ]; then
            log_msg "✅ 内存恢复正常: ${MEM_PCT}%"
            WARN_SENT=0
        fi
    fi

    sleep "$CHECK_INTERVAL"
done
