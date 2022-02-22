#!/bin/bash

prompt_err() {
  echo -e "COMMAND FAILED"
}

status() {
$1
if !( $? -eq 0 ); then
  prompt_err
  exit -1
fi
}

VEHICLE=$1

echo '||>>>RUNNING MAVPROXY<<<||'
if [ $VEHICLE == 'drone' ]; then
  status 'mavproxy.py --master=/dev/serial0 --baudrate 921600 --out=udp:192.168.18.131:14550 --out=udp:127.0.0.1:14551 --aircraft MyCopter'
elif [ $VEHICLE == 'rover' ]; then
  status 'mavproxy.py --master=/dev/serial0 --baudrate 57600 --out=udp:192.168.18.131:14550 --out=udp:127.0.0.1:14551 --aircraft MyRover'
elif [ $VEHICLE == 'simulation' ]; then
  # cd \Python27\Scripts
  status 'python mavproxy.py --master tcp:127.0.0.1:5760 --out=udp:127.0.0.1:14551 --out=udp:127.0.0.1:14550'
fi
