#!/usr/bin/env bash
# Keep a queue running on a preemptible Nebius VM (run locally): restart the VM when it is
# STOPPED (preemption), and relaunch the idempotent queue when it is not running, until its
# log shows DONE_PATTERN.
#   VM_ID=computeinstance-... QUEUE=experiments/s5_run.sh LOG=logs/s5_queue.log \
#     DONE_PATTERN="S5 DONE" bash experiments/nebius_supervise.sh
N=${NEBIUS:-~/.nebius/bin/nebius}
HOST_USER=${HOST_USER:-ssa}
REPO=${REPO:-/data/ssa}
: "${VM_ID:?}" "${QUEUE:?}" "${LOG:?}" "${DONE_PATTERN:?}"
while true; do
  state=$($N compute instance get --id "$VM_ID" --format jsonpath='{.status.state}' 2>/dev/null)
  if [ "$state" = STOPPED ]; then
    echo "$(date +%T) VM stopped (preempted?), starting"
    $N compute instance start --id "$VM_ID" >/dev/null 2>&1 || { sleep 120; continue; }
  elif [ "$state" = RUNNING ]; then
    ip=$($N compute instance get --id "$VM_ID" --format json | python3 -c \
      "import sys,json;print(json.load(sys.stdin)['status']['network_interfaces'][0]['public_ip_address']['address'].split('/')[0])")
    st=$(ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$HOST_USER@$ip" "cd $REPO;
      if grep -q '$DONE_PATTERN' $LOG 2>/dev/null; then echo done;
      elif pgrep -f '$QUEUE' >/dev/null; then echo running;
      else nohup bash $QUEUE >> $LOG 2>&1 < /dev/null & echo launched; fi" 2>/dev/null)
    echo "$(date +%T) $ip $st"
    [ "$st" = done ] && exit 0
  fi
  sleep 300
done
