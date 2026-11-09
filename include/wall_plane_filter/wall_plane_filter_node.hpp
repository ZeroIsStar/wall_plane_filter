#ifndef WALL_PLANE_FILTER__WALL_PLANE_FILTER_NODE_HPP_
#define WALL_PLANE_FILTER__WALL_PLANE_FILTER_NODE_HPP_

#include "wall_plane_filter/types.hpp"
#include "wall_plane_filter/wall_hough.hpp"
#include "wall_plane_filter/wall_track.hpp"

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "wall_plane_filter/msg/wall_plane.hpp"
#include "wall_plane_filter/msg/wall_plane_array.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace wall_plane_filter
{

class WallPlaneFilterNode : public rclcpp::Node
{
public:
  explicit WallPlaneFilterNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  using CloudT = pcl::PointCloud<pcl::PointXYZ>;

  // Latest preprocessed frame of one depth camera, waiting for the next fusion cycle.
  struct DepthSlot
  {
    // filtered / deviate: whole ROI in the wall work frame (per-camera output).
    CloudT filtered_world;
    CloudT deviate_world;
    // Wall-finding subset (no ground, near range, trusted noise), wall work frame.
    CloudT wall_world;
    Eigen::Vector3d origin_world{Eigen::Vector3d::Zero()};
    Eigen::Vector2d robot_xy{Eigen::Vector2d::Zero()};
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    std::chrono::steady_clock::time_point ready_steady{};
    bool fresh{false};
  };

  void loadParams();
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void depthCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg, size_t idx);
  void publishDepthCloud(size_t idx, const DepthSlot & slot);
  // Publish lidar filtered/deviate right after preprocessing (no fuse wait).
  void publishLidarCloud(const CloudT & filtered, const CloudT & deviate, const rclcpp::Time & stamp);
  // Fuse recent depth frames into an XY occupancy snapshot, then Hough.
  void wallFuseTick();
  // Publish geometric wall planes + LINE_LIST/TRIANGLE_LIST markers.
  void publishTrackedMarkers(
    const std::vector<WallLine> & lines, const std::vector<Eigen::Vector2d> & corners,
    const rclcpp::Time & stamp, const std::string & frame, bool depth);
  wall_plane_filter::msg::WallPlane toWallPlaneMsg(
    const WallLine & line, bool depth, double fallback_z_min, double fallback_z_max) const;
  void publishHoughDebug(
    const HoughResult & result, const rclcpp::Time & stamp,
    const std::string & frame, bool depth);
  // Keep the depth grid origin until the robot is about to leave it.
  void holdDepthGrid(const Eigen::Vector2d & robot, HoughConfig & config);
  void republishDepthClouds();
  void reportDepthTf();

  bool lookupIso(
    const std::string & to_frame, const std::string & from_frame, const rclcpp::Time & stamp,
    Eigen::Isometry3d & out);
  bool cloudToTarget(
    const sensor_msgs::msg::PointCloud2 & msg, CloudT & out, rclcpp::Time & stamp);
  // Transform an already-loaded cloud between frames at stamp (skip on TF fail).
  bool transformCloud(
    const CloudT & in, const std::string & to_frame, const std::string & from_frame,
    const rclcpp::Time & stamp, CloudT & out);
  // Frame used for the depth projection snapshot (map by default).
  std::string wallWorkFrame() const;
  // Frame of the output clouds and both wall markers (depth_wall_output_frame).
  std::string outputFrame() const;
  // Drop camera slots and validated lines when bag/TF time jumps backward.
  void clearWallCachesOnTimeJump(const rclcpp::Time & stamp);
  void filterRoi(CloudT & cloud) const;
  void voxelDownsample(CloudT & cloud) const;

  sensor_msgs::msg::PointCloud2 toMsg(
    const CloudT & cloud, const rclcpp::Time & stamp) const;
  sensor_msgs::msg::PointCloud2 toMsg(
    const CloudT & cloud, const rclcpp::Time & stamp, const std::string & frame) const;
  void publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & reliable,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & best_effort,
    const sensor_msgs::msg::PointCloud2 & msg, const char * label);

  Params params_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // One reliable and one best-effort subscription per topic. A publisher matches
  // only one of them, so the callback does not run twice.
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_compat_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_subs_;

  // Dual writers: RViz here uses Reliable; default PointCloud2 uses Best Effort.
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_filtered_pubs_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_deviate_pubs_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_filtered_pubs_be_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_deviate_pubs_be_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deviate_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub_be_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deviate_pub_be_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr lidar_wall_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr depth_wall_marker_pub_;
  rclcpp::Publisher<wall_plane_filter::msg::WallPlaneArray>::SharedPtr lidar_wall_plane_pub_;
  rclcpp::Publisher<wall_plane_filter::msg::WallPlaneArray>::SharedPtr depth_wall_plane_pub_;
  std::vector<std::string> depth_cam_names_;
  // Per-camera flag: 1 = feeds the depth projection.
  std::vector<char> depth_find_walls_;
  std::vector<DepthSlot> depth_slots_;
  // Per-camera TF lookups / failures since the last report.
  std::vector<size_t> depth_tf_total_;
  std::vector<size_t> depth_tf_fail_;
  std::chrono::steady_clock::time_point depth_tf_report_last_{std::chrono::steady_clock::now()};
  // Tracked map lines used by cullWallBand. Never the raw Hough of a previous tick.
  std::vector<WallLine> validated_depth_lines_;
  rclcpp::Time validated_depth_stamp_{0, 0, RCL_ROS_TIME};
  WallTracker depth_tracks_;
  WallTracker lidar_tracks_;
  Eigen::Vector2d depth_grid_origin_{Eigen::Vector2d::Zero()};
  bool depth_grid_valid_{false};
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr depth_grid_pub_, lidar_grid_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr depth_debug_pub_, lidar_debug_pub_;
  std::mutex depth_wall_mutex_;
  // Last stamp seen by the wall path (detect bag/TF time jumps).
  rclcpp::Time wall_path_stamp_{0, 0, RCL_ROS_TIME};
  // Set by depth callbacks when a cache changed; lidar has a separate projection.
  bool wall_data_dirty_{false};
  rclcpp::TimerBase::SharedPtr wall_fuse_timer_;

  rclcpp::Time last_stamp_{0, 0, RCL_ROS_TIME};
};

}  // namespace wall_plane_filter

#endif
