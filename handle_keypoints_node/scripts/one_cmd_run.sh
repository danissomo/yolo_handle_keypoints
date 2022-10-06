#!/bin/bash
container_name=keypoints_container
if ! [ "$( docker container inspect -f '{{.State.Status}}' $container_name )" == "running" ]; then 
    docker start $container_name
    docker exec -it $container_name /bin/bash -c "echo \"$HUSKY_IP cpr-fssl01\" >> /etc/hosts"
fi

init_cmd="source /opt/ros/noetic/setup.bash;
export ROS_MASTER_URI=http://$HUSKY_IP:11311; 
export ROS_IP=10.147.18.176; 
cd /home/keypoint_regression;
rm -rf ./build;
catkin_make;
source ./devel/setup.bash;
roslaunch handle_keypoints_node default.launch"
docker exec -it $container_name bash -c "$init_cmd"
