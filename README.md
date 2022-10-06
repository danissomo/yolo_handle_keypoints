# Door handle keypoint regression

Download weights from [here]()

## install 
```bash
mkdir catkin_ws
cd ./catkin_ws
mkdir src
catkin_make
mkdir src/handle_keypoints
git clone src/handle_keypoints
catkin_make
source devel/setup.bash
rosrun handle_keypoints_node start.sh
```
## run
```bash
rosrun handle_keypoints_node one_cmd_run.sh
```