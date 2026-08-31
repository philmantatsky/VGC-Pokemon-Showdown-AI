#!/bin/bash
# Apple Silicon variant of pretrain.sh: MPS instead of cuda:0, and uses the venv's python.
# Usage: ./pretrain_mac.sh [num_epochs]   (default 100, matching upstream)

run_id=1
port=8004
device="${DEVICE:-mps}"
num_epochs="${1:-100}"

start_showdown() {
    local port=$1
    (
        cd pokemon-showdown
        node pokemon-showdown start "$port" --no-security > /dev/null 2>&1 &
        echo $!
    )
}

echo "Starting Showdown server on port $port..."
showdown_pid=$(start_showdown "$port")
sleep 3
echo "Starting pretraining on device=$device for $num_epochs epochs..."
.venv/bin/python -m vgc_bench.pretrain \
    --run_id $run_id --port $port --device "$device" --num_epochs "$num_epochs"
exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "Pretraining process died with exit status $exit_status"
else
    echo "Pretraining process finished!"
fi
kill $showdown_pid 2>/dev/null
