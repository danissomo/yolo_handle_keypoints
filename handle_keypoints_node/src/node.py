#!/usr/bin/env python3
import math

import cv2
import numpy as np
import rospy
import tf
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, Pose2D
from handle_keypoints_msgs.srv import FindHandle, FindHandleResponse
from loguru import logger
from message_filters import Subscriber, TimeSynchronizer
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from vision_msgs.msg import BoundingBox2D
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info

from ros_parser import make_parser
from torch_utils import DOOR_HANDLE_CLASSES, Predictor


class DoorHandleHandler:
    def __init__(self, predictor, regressor):
        self.predictor = predictor
        self.regressor = regressor

        self.listner = tf.TransformListener()
        self.serv = rospy.Service("FindHandle", FindHandle, self.image_callback)
        # self.synchronizer = TimeSynchronizer(
        #     [
        #         Subscriber('/realsense_back/color/image_raw/compressed', CompressedImage),
        #         Subscriber('/realsense_back/aligned_depth_to_color/image_raw', Image),
        #         Subscriber('/realsense_back/color/camera_info', CameraInfo),
        #     ],
        #     queue_size=30,
        # )
        # self.synchronizer.registerCallback(self.image_callback)
        #self.pointArrayPub = rospy.Publisher("/handle/points_yolo", PoseArray, queue_size=10)

        self.bridge = CvBridge()
        self.count = 0



    def depth_to_pc(self, depth, keypoints_coordinates, camera_info):
        depth = depth / 1000
        fx, _, cx, _, fy, cy, *_ = camera_info
        height, width = depth.shape
        
        for i in range(len(keypoints_coordinates)):
            def meanZ(x, y, depth):
                return np.mean(np.array([ 
                                depth[int(dy + y) ][int(dx + x)] 
                                for dx, dy in zip( (1, 1, -1, -1), (1, -1, 1, -1))]))
            def project(x, y, z):
                return np.array([
                    z*(x - cx) / fx,
                    z*(y - cy) / fy,
                    z
                ])
            
            def keypointToPose(point, z):
                pose = Pose()
                pose.position.z = z
                pose.position.x = pose.position.z * (point[0] - cx) / fx
                pose.position.y = pose.position.z * (point[1] - cy) / fy
                return pose

            
            handlePointsArray = PoseArray()
            basePoint1 = keypoints_coordinates[i]['keypoints'][2]
            basePoint2 = keypoints_coordinates[i]['keypoints'][3]
            check_dims =  0 <= basePoint1[0] <= width \
                     and 0 <= basePoint1[1] <= height
            if not check_dims:
                continue
            try:
                p =  [ project(
                    basePoint1[0] + dx,
                    basePoint2[1] + dy,
                    meanZ(basePoint1[0] + dx, basePoint2[1] + dy, depth)
                ) for dx, dy in [(0,50), (10, 50), (0, 60)]]

                normal = np.cross(
                    p[2] - p[0],
                    p[1] - p[0]
                )
                if math.isnan( np.linalg.norm(normal)) or math.isinf(np.linalg.norm(normal)):
                    continue 
                normal /= np.linalg.norm(normal)
            except Exception as e:
                rospy.logdebug(f"catched except {e}")
                continue
            handlePointsArray.poses = [
                 keypointToPose(p, meanZ(p[0], p[1], depth))
                for p in [basePoint1]*2 + [basePoint2]*2
            ]
            
            handlePointsArray.poses[ 0].position.x += 0.04*normal[0]
            handlePointsArray.poses[ 0].position.y += 0.04*normal[1]
            handlePointsArray.poses[ 0].position.z += 0.04*normal[2]
            handlePointsArray.poses[-1].position.x += 0.04*normal[0]
            handlePointsArray.poses[-1].position.y += 0.04*normal[1]
            handlePointsArray.poses[-1].position.z += 0.04*normal[2]
            handlePointsArray.header.frame_id = "rs_camera"
            handlePointsArray.header.stamp = rospy.get_rostime()
            
            return FindHandleResponse(
                handlePointsArray,
                BoundingBox2D(
                    center = Pose2D(keypoints_coordinates[i]['bbox'][0], keypoints_coordinates[i]['bbox'][1], 0),
                    size_x = keypoints_coordinates[i]['bbox'][2] - keypoints_coordinates[i]['bbox'][0],
                    size_y = keypoints_coordinates[i]['bbox'][3] - keypoints_coordinates[i]['bbox'][1]
                )
            )
        return FindHandleResponse()

    #def image_callback( self, image_msg : Image, depth_msg : Image, camera_info_msg : CameraInfo):       
    def image_callback( self, *smsthing ):
        image_msg : CompressedImage = rospy.wait_for_message("/realsense_back/color/image_raw/compressed", CompressedImage)
        depth_msg : Image = rospy.wait_for_message('/realsense_back/aligned_depth_to_color/image_raw', Image)
        camera_info_msg : CameraInfo = rospy.wait_for_message('/realsense_back/color/camera_info', CameraInfo)
   
        np_img_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_img = cv2.imdecode(np_img_arr, cv2.IMREAD_COLOR)
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")

        assert cv_img.shape[:2] == cv_depth.shape, f'Shapes of depth and image should be the same{cv_img.shape[:2]}, {cv_depth.shape}'
        
        outputs, img_info = self.predictor.inference(cv_img)
        result_det = self.predictor.visual(outputs[0], img_info, self.predictor.confthre)
        try:
            pose_results, returned_outputs = inference_top_down_pose_model(
                self.regressor,
                img_info['raw_img'],
                result_det,
                bbox_thr=0.0,
                format='xyxy',
                dataset='TopDownCocoDataset',
            )
        except:
            return FindHandleResponse()   
        return self.depth_to_pc(
                    depth=cv_depth,
                    keypoints_coordinates=pose_results,
                    camera_info=camera_info_msg.K,
                )
        
        self.count += 1
        
    def run(self):
        rospy.spin()

    
def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
        
    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device_det == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        ckpt_file = args.ckpt_det
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    trt_file = None
    decoder = None

    predictor = Predictor(
        model, exp, DOOR_HANDLE_CLASSES, trt_file, decoder,
        args.device_det, args.fp16, args.legacy,
    )

    regressor = init_pose_model(
        args.exp_file_reg, args.ckpt_reg, args.device_reg,
    )

    process_ros = DoorHandleHandler(
            predictor=predictor,
            regressor=regressor,
        )
        
    process_ros.run()


if __name__ == "__main__":
    rospy.init_node('keypoints_detection', log_level=rospy.DEBUG)
    args = make_parser()
    exp = get_exp(args.exp_file_det, args.name)

    main(exp, args)
