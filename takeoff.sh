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

FLIGHTSCRIPT=$1
cd /home/pi/Desktop

echo 'Connect Mission Planner before preceding'
echo
echo 'READY TO EXECUTE TAKEOFF SCRIPT...' $FLIGHTSCRIPT
echo "Type 'takeoff' when ready to arm and takeoff or 'q' to cancel..."
read -p '>>> ' TAKEOFFVAR
if [ $TAKEOFFVAR == 'takeoff' ]; then
  echo '||>>>RUNNING TAKEOFF SCRIPT<<<||'
  # while ! (python scriptname.py --connect udp:127.0.0.1:14551 -eq 0)
  # while ! (python AdvancedDay1.py -eq 0) # PROB WON'T NEED
  # do
  #     echo Attempting to connect
  #     sleep 1
  # done
  status 'python '$FLIGHTSCRIPT' --connect udp:127.0.0.1:14551'
elif [ $TAKEOFFVAR == 'cancel' ]; then
  exit -1
fi