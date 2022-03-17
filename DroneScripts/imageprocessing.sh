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

echo
echo 'READY TO EXECUTE PATHFINDING AND MAPPING ON IMAGES'
echo
echo 'Review drone images and image data.'
echo 'The pathfinding and mapping algorithms will process the images'
echo 'and compute the best path for the rover to reach its destination.'
# echo "The resulting navigation script is saved as 'RoverNavScript.py'."
echo
echo "Type 'execute' when ready to process drone images or 'q' to cancel..."

read -p '>>> ' PROCESS_IMG_VAR

while [[ $PROCESS_IMG_VAR != 'execute' && $PROCESS_IMG_VAR != 'q' ]]
    do
        echo 'Error: invalid input'
        read -p '>>> ' PROCESS_IMG_VAR
    done

if [ $PROCESS_IMG_VAR == 'execute' ]; then
    echo '||>>>RUNNING PATHFINDING AND MAPPING ALGORITHMS ON IMAGES<<<||'
    cd ..
    python PathfindingAndMapping.py

    # print image choices
    # ask user to pick one image and its corresponding gps path to create a navscript for the rover
    # run script that creates navscript from gps coordinates
fi