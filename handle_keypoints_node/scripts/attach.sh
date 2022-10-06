#!/bin/bash
container_name=keypoints_container
if ! [ "$( docker container inspect -f '{{.State.Status}}' $container_name )" == "running" ]; then 
    docker start $container_name
    docker exec -it $container_name /bin/bash -c "echo \"$HUSKY_IP cpr-fssl01\" >> /etc/hosts"
fi

docker exec -it keypoints_container /bin/bash -c "export HUSKY_IP=$HUSKY_IP; cd /home/keypoint_regression; /bin/bash --init-file /home/keypoint_regression/src/handle_keypoints/handle_keypoints_node/scripts/init_file.sh"