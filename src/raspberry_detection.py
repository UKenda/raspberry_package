#!/usr/bin/python3


from importlib.machinery import BYTECODE_SUFFIXES
import rospy
import cv2
import numpy as np
import rospkg
import ros_numpy
import struct
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Header
from sensor_msgs import point_cloud2

cv_bridge = CvBridge()
rospack = rospkg.RosPack()


fx = 905.133  
fy = 905.855
cx = 639.099
cy = 359.523
tx = 0.00103
ty = -0.0144596
tz = 0.000142495

K = np.array([[fx, 0, cx, 0],
              [0, fy, cy, 0],
              [0, 0 , 1, 0]])

Rt = np.array([[1, 0, 0, tx],
               [0, 1, 0 ,ty],
               [0 ,0 ,1 ,tz],
               [0 ,0 ,0 ,1]])

def project(pt):
    pt_mat = np.array([[pt[0]],
                       [pt[1]],
                       [pt[2]],
                       [1]])
    
    uvw = K @ Rt @ pt_mat
    return (int(uvw[0]/uvw[2]), int(uvw[1]/uvw[2]))

def build_model(is_cuda):
    YOLO_VERSION = "v4-raspberry"
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

def main():
    rospy.init_node("yolo_node")

    pub_yolo_image = rospy.Publisher("/yolo/image",Image,queue_size=1)
    #pub_yolo_depth = rospy.Publisher("/yolo/depth",Image,queue_size=1)
    pub_pc2 = rospy.Publisher("data/point_cloud2", PointCloud2, queue_size=2)
    pub_pc2_detection = rospy.Publisher("costom/point_cloud2", PointCloud2, queue_size=2)
    marker_arr_pub = rospy.Publisher("/raspberry/markers", MarkerArray)
    marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=2)
    i=0    
    markerArray = MarkerArray()
    while not rospy.is_shutdown():
    
        print("New image")
        pointCloudMsg = (rospy.wait_for_message("/camera/depth/color/points", PointCloud2, timeout=None))
        if (pointCloudMsg.width * pointCloudMsg.height) == 0:
            return #return if the cloud is not dense!

        pc = ros_numpy.numpify(pointCloudMsg)

        shape = pc.shape + (4, )
        points = np.zeros(shape) 
        points[..., 0] = pc['x']
        points[..., 1] = pc['y']
        points[..., 2] = pc['z']
        points[..., 3] = pc['rgb']
        cv_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        for point in points:
            image_point = project(point)
            # Draw the point on the image
            rgb_val_ = struct.unpack('I', struct.pack('f', point[3]))[0]
            r = (rgb_val_ >> 16) & 0x0000ff
            g = (rgb_val_ >> 8) & 0x0000ff
            b = rgb_val_ & 0x0000ff
            cv2.circle(cv_image, image_point, 3, (b,g,r), -1)
        
        
        

        classIds, confidences, boxes = model.detect(cv_image, .2, .4)


        anotatedBoxes = []
        i=0
        
        for (classid, confidence, box) in zip(classIds, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            if confidence > 0.50:
                if classid == 0:
                    anotatedBoxes.append(box)


                cv2.rectangle(cv_image, box, color, 2)
                cv2.rectangle(cv_image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                cv2.putText(cv_image, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0))
                cv2.putText(cv_image, str(np.round(confidence,2)),(box[0], box[1] - 0), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0))




        num_boxes = 0
        for box in anotatedBoxes:
            
            num_boxes += 1

        
        centerRaspberry = np.zeros((num_boxes,4)) 
        customPoint = []
        r = int(105 )
        g = int(0)
        b = int(143)
        a = int(255)
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        
        for point in points:
            image_point = project(point)  
            i = 0    
            for box in anotatedBoxes:     
                
                   
                #for box in anotatedBoxes:
                if(image_point[0] > box[0] and image_point[0] < box[0]+box[2] and 
                image_point[1] > box[1] and image_point[1] < box[1]+box[3] ):
                    #print(cameraCordinates)
                    '''Use when drawing pc2 only raspberries'''
                    centerRaspberry [i] +=[point[0],point[1],point[2],1] 
                i =i+1

                ''' customPoint.append([point[0],point[1],point[2],rgb])
                    #print("hi")
                
                
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                # PointField('rgb', 12, PointField.UINT32, 1),
                PointField('rgba', 12, PointField.UINT32, 1),
                ]
        #print(customPoint)
        header = Header()
        header.frame_id = "camera_depth_optical_frame"
        pc2 = point_cloud2.create_cloud(header, fields, customPoint)
        pc2.header.stamp = rospy.Time.now()
        pub_pc2_detection.publish(pc2)
        '''
        i=int(0)
        for m in markerArray.markers:
            marker[0].action = marker[0].DELETEALL
        marker = [Marker() for i in range(num_boxes)]
        i=int(0)
        for raspbery in centerRaspberry:
            raspbery[0] = raspbery[0]/raspbery[3]
            raspbery[1] = raspbery[1]/raspbery[3]
            raspbery[2] = raspbery[2]/raspbery[3]
            raspbery[3] = 0
            marker[i].id = i
            marker[i].header.frame_id = "camera_depth_optical_frame"
            marker[i].header.stamp = rospy.Time.now()
            marker[i].action = marker[i].ADD
            marker[i].type = marker[i].SPHERE

            marker[i].scale.x = 0.04
            marker[i].scale.y = 0.04
            marker[i].scale.z = 0.04
            marker[i].color.r = 0.0
            marker[i].color.g = 0.0
            marker[i].color.b = 1.0
            marker[i].color.a = 0.5
            marker[i].pose.position.x = raspbery[0]
            marker[i].pose.position.y = raspbery[1]
            marker[i].pose.position.z = raspbery[2]
            markerArray.markers.append(marker[i])
            i += 1
        marker_arr_pub.publish(markerArray)

        pub_pc2.publish(pointCloudMsg)
        pub_yolo_image.publish(cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")) #publish our cloud image

if __name__ == "__main__":
    main()