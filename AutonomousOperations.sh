#!/bin/bash

echo Welcome to Drover

VEHICLE='simulation'
FLIGHTSCRIPT='SimpleTakeOffLand.py'

# On Raspberry pi (linux)
# gnome-terminal -- "cd /directory && ./mavproxyset.sh" &
# gnome-terminal --noclose "cd /directory && ./BashTesting.sh" $FLIGHTSCRIPT &
# exit # don't know what exit does or if it's necessary

# On windows
# xterm -e -hold "cd /directory && ./mavproxyset.sh simulation" &
# xterm -e -hold "cd /directory && ./BashTesting.sh SimpleTakeOffLand.py" &
# exit # don't know what exit does or if it's necessary

#On Mac
osascript -e 'tell app "Terminal"
  do script "cd /Users/charlesjiang/Downloads && ./mavproxysetup.sh simulation"
  do script "cd /Users/charlesjiang/Downloads && ./BashTesting.sh SimpleTakeOffLand.py"
end tell'

# if vehicle is drone:
# Access photo and run astar
# Run pixel to coordinates

# Send coordinate info to computer on ground
