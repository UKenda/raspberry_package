#!/usr/bin/python3


import rospy
import cv2
from cv_bridge import CvBridge 
import os
import rospkg
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image
from foxglove_msgs.msg import ImageMarkerArray
from visualization_msgs.msg import ImageMarker
from geometry_msgs.msg import Point
import numpy as np
cv_bridge = CvBridge()
rospack = rospkg.RosPack()





def build_model(is_cuda):
    YOLO_VERSION = "v4-raspberry-tiny"
    net = cv2.dnn.readNet(rospack.get_path('raspberry_package')+"/yolo_config/yolo" + YOLO_VERSION + ".weights", rospack.get_path('raspberry_package')+"/yolo_config/yolo" + YOLO_VERSION + ".cfg")
    #net = cv2.dnn.readNet("config_files/yolov3_custom_last.weights", "config_files/yolov3_testing.cfg")
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1./255, swapRB=True)
    return model


def load_classes():
    class_list = []
    with open(rospack.get_path('raspberry_package')+"/yolo_config/raspberry_classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

colors = [(0, 0, 255), (0, 255, 0)]
model = build_model("cuda")
class_list = load_classes()
pub_yolo_image = rospy.Publisher("/yolo/image",Image,queue_size=1)

def color_image_callback( msg: Image):
    img = cv2.cvtColor(cv_bridge.imgmsg_to_cv2(msg), cv2.COLOR_RGB2BGR)
    classIds, confidences, boxes = model.detect(img, .2, .4)
    for (classid, confidence, box) in zip(classIds, confidences, boxes):
        color = colors[int(classid) % len(colors)]
        if confidence > 0.45:
            cv2.rectangle(img, box, color, 2)
            cv2.rectangle(img, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(img, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0))
            cv2.putText(img, str(np.round(confidence,2)),(box[0], box[1] - 0), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0))
    pub_yolo_image.publish(cv_bridge.cv2_to_imgmsg(img, "bgr8"))

def main():
    rospy.init_node("yolo_node")
    rospy.Subscriber("/camera/color/image_raw", Image, color_image_callback, queue_size=1)

    yolo_detection = None
    
    #pub_ripe_raspberry = rospy.Publisher("/yolo/ripe_raspberry", ImageMarkerArray, queue_size=1)
    #pub_unripe_raspberry = rospy.Publisher("/yolo/unripe_raspberry", ImageMarkerArray, queue_size=1)

    rospy.spin()


if __name__ == "__main__":
    main()