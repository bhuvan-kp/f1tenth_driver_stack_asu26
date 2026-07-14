#!/bin/bash
set -e

# Source ROS2 base. NOTE: we deliberately do NOT try to source
# /f1tenth_ws/install/setup.bash here -- since package source is
# bind-mounted rather than baked into the image, that file doesn't exist
# until colcon build has run at least once, which happens as part of the
# container's CMD (see Dockerfile / docker-compose.yml).
source /opt/ros/humble/setup.bash

exec "$@"
