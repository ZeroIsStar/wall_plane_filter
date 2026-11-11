#ifndef WALL_PLANE_FILTER__WALL_PLANE_FILTER_NODE_HPP_
#define WALL_PLANE_FILTER__WALL_PLANE_FILTER_NODE_HPP_

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "wall_plane_filter/msg/wall_plane_array.hpp"
#include "wall_plane_filter/types.hpp"

namespace wall_plane_filter
{

class WallPlaneFilterNode : public rclcpp::Node
{
public:
  explicit WallPlaneFilterNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  using CloudT = pcl::PointCloud<pcl::PointXYZ>;

  struct DepthSlot
  {
    CloudT filtered_world;
    CloudT deviate_world;
    Eigen::Vector2d robot_xy{Eigen::Vector2d::Zero()};
    Eigen::Vector2d sensor_xy{Eigen::Vector2d::Zero()};
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  };

  void loadParams();
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void depthCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg, size_t idx);
  void processDepthWalls(
    size_t idx, const CloudT & wall_world, const Eigen::Vector2d & robot_xy,
    const rclcpp::Time & stamp);
  void publishLidarWalls(
    const std::vector<WallLine> & lines, const rclcpp::Time & stamp, const std::string & frame);
  void publishDepthCloud(size_t idx, const DepthSlot & slot);
  void publishCamWalls(
    size_t idx, const std::vector<WallLine> & lines, const rclcpp::Time & stamp,
    const std::string & frame);
  void reportDepthTf();
  bool lookupIso(
    const std::string & to_frame, const std::string & from_frame, const rclcpp::Time & stamp,
    Eigen::Isometry3d & out);
  bool transformCloud(
    const CloudT & in, const std::string & to_frame, const std::string & from_frame,
    const rclcpp::Time & stamp, CloudT & out);
  std::string wallWorkFrame() const;
  std::string outputFrame() const;
  void clearWallCachesOnTimeJump(const rclcpp::Time & stamp);
  bool acceptStampOnce(int key, int64_t stamp_ns);
  sensor_msgs::msg::PointCloud2 toMsg(
    const CloudT & cloud, const rclcpp::Time & stamp, const std::string & frame) const;
  void publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & reliable,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & best_effort,
    const sensor_msgs::msg::PointCloud2 & msg);

  Params params_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_be_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_filtered_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_filtered_pub_be_;
  rclcpp::Publisher<wall_plane_filter::msg::WallPlaneArray>::SharedPtr lidar_wall_plane_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr lidar_wall_marker_pub_;
  WallTracker lidar_tracks_;
  std::mutex lidar_wall_mutex_;

  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_subs_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_filtered_pubs_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_deviate_pubs_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_filtered_pubs_be_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_deviate_pubs_be_;
  std::vector<rclcpp::Publisher<wall_plane_filter::msg::WallPlaneArray>::SharedPtr>
    depth_cam_plane_pubs_;
  std::vector<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr>
    depth_cam_marker_pubs_;

  std::vector<std::string> depth_cam_names_;
  std::vector<char> depth_find_walls_;
  std::vector<DepthSlot> depth_slots_;
  std::vector<WallTracker> depth_cam_tracks_;
  /** Last finalized (clamped + corner-snapped) walls per camera for multi-cam fuse. */
  std::vector<std::vector<WallLine>> depth_cam_published_;
  std::vector<size_t> depth_tf_total_;
  std::vector<size_t> depth_tf_fail_;
  std::chrono::steady_clock::time_point depth_tf_report_last_{std::chrono::steady_clock::now()};
  std::mutex depth_tf_mutex_;
  std::mutex depth_wall_mutex_;
  std::vector<WallLine> validated_depth_lines_;
  rclcpp::Time validated_depth_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time wall_path_stamp_{0, 0, RCL_ROS_TIME};
  std::mutex dedup_mutex_;
  std::deque<std::pair<int, int64_t>> recent_stamps_;
  static constexpr size_t kDedupHistory = 64;
};

}  // namespace wall_plane_filter

#endif  // WALL_PLANE_FILTER__WALL_PLANE_FILTER_NODE_HPP_
