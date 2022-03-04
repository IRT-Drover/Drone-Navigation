#!/bin/bash

echo Welcome to Drover

# run to allow permission to run:
# ls -l filename.sh
# chmod 755 filename.sh

VEHICLE='rover'
FLIGHTSCRIPT='WayPointNavScript.py'

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

# On Raspberry pi (linux)
lxterminal --command "cd /home/pi/Desktop && ./mavproxysetup.sh "$VEHICLE &
lxterminal --command "cd /home/pi/Desktop && ./takeoff.sh "$FLIGHTSCRIPT &

#On Mac
# osascript -e 'tell app "Terminal"
#   do script "cd /Users/charlesjiang/Downloads && ./mavproxysetup.sh drone"
#   do script "cd /Users/charlesjiang/Downloads && ./takeoff.sh SimpleTakeOffLand.py"
# end tell'

if [ $VEHICLE -eq 'drone' ]; then
  # # Access photo and run astar
  # # Save pixel coordinate data and image with path drawn on it
  # status 'python astar.py'

  # # Run pixel to coordinates
  # # Save coordinate data to file
  # status 'python PixelstoCoordinates.py'

# Send coordinate data to computer on ground


# bad interpreter error because of differing carriage return characters:
# sed -i -e 's/\r$//' file.sh