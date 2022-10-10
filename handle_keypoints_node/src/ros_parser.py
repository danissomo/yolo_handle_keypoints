import rospy
def make_parser():
    class Args:
        def __init__(self) -> None: 
            self.demo = rospy.get_param(
                "~demo",
                default="image",
            )
            self.experiment_name = rospy.get_param(
                "~experiment-name",
                default=None,
            )
            self.path = rospy.get_param(
                "~path",
                default="./assets/dog.jpg",
            )
            self.camid = rospy.get_param(
                "~camid",
                default=0,
            )
            self.save_result = rospy.get_param(
                "~save_result",
                default=False
            )
            #detector config
            self.name = rospy.get_param(
                "~name",
                default=None,
            )
            self.exp_file_det = rospy.get_param(
                 "~exp_file_det",
                default=None,
            )
            self.ckpt_det = rospy.get_param(
                 "~ckpt_det",
                default=None,
            )
            self.device_det = rospy.get_param(
                "~device_det",
                default="cpu",
            )
            self.conf = rospy.get_param(
                "~conf",
                default=0.3,
            )
            self.nms = rospy.get_param(
                "~nms",
                default=0.3,
            )
            self.tsize = rospy.get_param(
                "~tsize",
                default=None,
            )
            self.fp16 = rospy.get_param(
                "~fp16",
                default=False,
            )
            self.legacy = rospy.get_param(
                "~legacy",
                default=False,
            )
            self.fuse = rospy.get_param(
                "~fuse",
                default=False,
            )
            self.trt = rospy.get_param(
                "~trt",
                default=False,
            )
            #keypoint config
            self.ckpt_reg = rospy.get_param(
                 "~ckpt_reg",
                default=None,
            )
            self.exp_file_reg = rospy.get_param(
                 "~exp_file_reg",
                default=None,
            )
            self.device_reg = rospy.get_param(
                "~device_reg",
                default="cpu",
            )
            
    a = Args()
    rospy.loginfo(a.exp_file_det)
    return  a
