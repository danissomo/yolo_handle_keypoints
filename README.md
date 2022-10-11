# Door handle keypoint regression



## used env vars
HUSKY_IP=10.147.18.169

## requirement
1. [vision_msgs](https://github.com/ros-perception/vision_msgs)

## install 
```bash
mkdir catkin_ws
cd ./catkin_ws
mkdir src
catkin_make
mkdir src/handle_keypoints
git clone --recurse-submodules https://github.com/danissomo/yolo_handle_keypoints.git src/handle_keypoints
catkin_make
source devel/setup.bash
rosrun handle_keypoints_node start.sh
```
Download weights from [here](https://disk.yandex.ru/d/bbVGap6W7AGOWg) to ```handle_keypoints/handle_keypoints_node/src/weights```
## run
```bash
rosrun handle_keypoints_node one_cmd_run.sh
```