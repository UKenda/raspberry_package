// ROS core
#include <ros/ros.h>
//Image message
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
//pcl::toROSMsg
#include <opencv2/opencv.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cv_bridge/cv_bridge.h>


float fx = 905.133;  
float fy = 905.855;
float cx = 639.099;
float cy = 359.523;
float tx = 0.00103;
float ty = -0.0144596;
float tz = 0.000142495;

cv::Mat K = (cv::Mat_<double>(3,4) << fx, 0, cx, 0,
                                      0, fy, cy, 0,
                                      0, 0 , 1, 0);
cv::Mat Rt = (cv::Mat_<double>(4,4) << 1, 0, 0, tx,
                                       0, 1, 0, ty,
                                       0, 0, 1, tz,
                                       0, 0, 0, 1);


cv::Point2d project(const cv::Point3d& pt)
{
    cv::Mat pt_mat = (cv::Mat_<double>(4,1) << pt.x,
                                              pt.y,
                                              pt.z,
                                              1);
    
    cv::Mat uvw = K * Rt * pt_mat;
    return cv::Point2d(uvw.at<double>(0)/uvw.at<double>(2),
                       uvw.at<double>(1)/uvw.at<double>(2));
}



class PointCloudToImage {
public:
  void
  cloud_cb (const sensor_msgs::PointCloud2& pointCloudMsg) {
    if ((pointCloudMsg.width * pointCloudMsg.height) == 0)
      return; //return if the cloud is not dense!
    // Convert the sensor_msgs::PointCloud2 message to a pcl::PointCloud
    pcl::PointCloud<pcl::PointXYZRGB> pointCloud;
    pcl::fromROSMsg(pointCloudMsg, pointCloud);
    cv::Mat cv_image = cv::Mat::zeros(720, 1280, CV_8UC3);
    for (const auto& point : pointCloud)
    {
        cv::Point3d world_point(point.x, point.y, point.z);
        cv::Point2d image_point = project(world_point);
        // Draw the point on the image
        uint32_t rgb_val_;
        memcpy(&rgb_val_, &(point).rgb,sizeof(uint32_t));
        uint8_t r,g,b;
        r= (rgb_val_ >>16) & 0x0000ff;
        g= (rgb_val_ >>8) & 0x0000ff;
        b= (rgb_val_) & 0x0000ff;
         cv::circle(cv_image, image_point, 3, cv::Scalar(b, g, r), -1);
    }
    cv_bridge::CvImage cvi;
    cvi.header.stamp = ros::Time::now();
    cvi.header.frame_id = "image";
    cvi.encoding = "bgr8";
    cvi.image = cv_image;
    cvi.toImageMsg(image_);
    //printf("array is: %f",array[0]);
    printf("image dimensions  %dx%d\n",image_.width,image_.height);
    image_pub_.publish (image_); //publish our cloud image
  }


  PointCloudToImage () {
    sub_ = nh_.subscribe ("/camera/depth/color/points", 30,
                          &PointCloudToImage::cloud_cb, this);
    image_pub_ = nh_.advertise<sensor_msgs::Image> ("/camera/depth/color/image", 30);

    //print some info about the node
    std::string r_ct = nh_.resolveName (cloud_topic_);
    std::string r_it = nh_.resolveName (image_topic_);
    ROS_INFO_STREAM("Listening for incoming data on topic " << r_ct );
    ROS_INFO_STREAM("Publishing image on topic " << r_it );
  }
private:
  ros::NodeHandle nh_;
  sensor_msgs::Image image_; //cache the image message
  std::string cloud_topic_; //default input
  std::string image_topic_; //default output
  ros::Subscriber sub_; //cloud subscriber
  ros::Publisher image_pub_; //image message publisher
};

int
main (int argc, char **argv)
{  
  ros::init (argc, argv, "convert_pointcloud_to_image");
  PointCloudToImage pci; //this loads up the node
  ros::spin (); //where she stops nobody knows
  return 0;
}