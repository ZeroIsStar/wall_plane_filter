#ifndef WALL_PLANE_FILTER__WALL_PLANE_FILTER_NODE_HPP_
#define WALL_PLANE_FILTER__WALL_PLANE_FILTER_NODE_HPP_

#include "wall_plane_filter/types.hpp"
#include "wall_plane_filter/wall_hough.hpp"
#include "wall_plane_filter/wall_track.hpp"

#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
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
#include "wall_plane_filter/msg/elevator_door_status.hpp"
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

  // Latest preprocessed frame of one depth camera.
  struct DepthSlot
  {
    CloudT filtered_world;
    CloudT deviate_world;
    Eigen::Vector2d robot_xy{Eigen::Vector2d::Zero()};
    // Camera origin in wall work frame (for wall-band noise model).
    Eigen::Vector2d sensor_xy{Eigen::Vector2d::Zero()};
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
  };

  void loadParams();
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void depthCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg, size_t idx);
  void publishDepthCloud(size_t idx, const DepthSlot & slot);
  void publishLidarCloud(const CloudT & filtered, const CloudT & deviate, const rclcpp::Time & stamp);
  // Run Hough+track for one depth camera and publish planes immediately.
  void processDepthWalls(
    size_t idx, const CloudT & wall_world, const Eigen::Vector2d & robot_xy,
    const rclcpp::Time & stamp);
  // Publish geometric wall planes + LINE_LIST/TRIANGLE_LIST markers.
  void publishTrackedMarkers(
    const std::vector<WallLine> & lines, const std::vector<Eigen::Vector2d> & corners,
    const rclcpp::Time & stamp, const std::string & frame, bool depth);
  void publishDepthCamPlanes(
    size_t idx, const std::vector<WallLine> & lines, const rclcpp::Time & stamp,
    const std::string & frame);
  wall_plane_filter::msg::WallPlane toWallPlaneMsg(
    const WallLine & line, bool depth, double fallback_z_min, double fallback_z_max) const;
  void publishHoughDebug(
    const HoughResult & result, const rclcpp::Time & stamp,
    const std::string & frame, bool depth);
  void holdDepthGrid(size_t idx, const Eigen::Vector2d & robot, HoughConfig & config);
  std::vector<WallLine> collectDepthTrackedLines(double stamp_sec);
  void reportDepthTf();
  // Point cloud must be in target_frame (base_link). Updates vote FSM and publishes.
  void updateElevatorDoor(const CloudT & cloud_base, const rclcpp::Time & stamp);

  bool lookupIso(
    const std::string & to_frame, const std::string & from_frame, const rclcpp::Time & stamp,
    Eigen::Isometry3d & out);
  bool cloudToTarget(
    const sensor_msgs::msg::PointCloud2 & msg, CloudT & out, rclcpp::Time & stamp);
  bool transformCloud(
    const CloudT & in, const std::string & to_frame, const std::string & from_frame,
    const rclcpp::Time & stamp, CloudT & out);
  std::string wallWorkFrame() const;
  std::string outputFrame() const;
  void clearWallCachesOnTimeJump(const rclcpp::Time & stamp);
  void filterRoi(CloudT & cloud) const;
  void voxelDownsample(CloudT & cloud) const;

  // Dual-QoS dedup: key -1 = lidar, >=0 = depth camera index.
  bool acceptStampOnce(int key, int64_t stamp_ns);

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

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_compat_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_subs_;

  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_filtered_pubs_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_deviate_pubs_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_filtered_pubs_be_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> depth_deviate_pubs_be_;
  std::vector<rclcpp::Publisher<wall_plane_filter::msg::WallPlaneArray>::SharedPtr> depth_cam_plane_pubs_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deviate_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub_be_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deviate_pub_be_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr lidar_wall_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr depth_wall_marker_pub_;
  rclcpp::Publisher<wall_plane_filter::msg::WallPlaneArray>::SharedPtr lidar_wall_plane_pub_;
  rclcpp::Publisher<wall_plane_filter::msg::WallPlaneArray>::SharedPtr depth_wall_plane_pub_;
  rclcpp::Publisher<wall_plane_filter::msg::ElevatorDoorStatus>::SharedPtr elevator_door_pub_;
  std::vector<std::string> depth_cam_names_;
  std::vector<char> depth_find_walls_;
  std::vector<DepthSlot> depth_slots_;
  std::vector<size_t> depth_tf_total_;
  std::vector<size_t> depth_tf_fail_;
  std::chrono::steady_clock::time_point depth_tf_report_last_{std::chrono::steady_clock::now()};
  std::mutex depth_tf_mutex_;
  // Union of per-camera tracked map lines for cullWallBand.
  std::vector<WallLine> validated_depth_lines_;
  rclcpp::Time validated_depth_stamp_{0, 0, RCL_ROS_TIME};
  std::vector<WallTracker> depth_cam_tracks_;
  WallTracker lidar_tracks_;
  // Last non-empty MarkerArray for coasting when the current observation is empty.
  visualization_msgs::msg::MarkerArray last_depth_wall_markers_;
  visualization_msgs::msg::MarkerArray last_lidar_wall_markers_;
  wall_plane_filter::msg::WallPlaneArray last_depth_wall_planes_;
  wall_plane_filter::msg::WallPlaneArray last_lidar_wall_planes_;
  // Per-camera Hough grid origins (follow_sensor).
  std::vector<Eigen::Vector2d> depth_grid_origins_;
  std::vector<char> depth_grid_valid_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr depth_grid_pub_, lidar_grid_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr depth_debug_pub_, lidar_debug_pub_;
  std::mutex depth_wall_mutex_;
  rclcpp::Time wall_path_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_stamp_{0, 0, RCL_ROS_TIME};
  std::mutex stamp_mutex_;
  std::mutex dedup_mutex_;
  std::deque<std::pair<int, int64_t>> recent_stamps_;
  static constexpr size_t kDedupHistory = 64;

  // Elevator door vote FSM (shared across lidar/depth observations).
  uint8_t elevator_door_state_{wall_plane_filter::msg::ElevatorDoorStatus::UNKNOWN};
  uint8_t elevator_door_pending_{wall_plane_filter::msg::ElevatorDoorStatus::UNKNOWN};
  int elevator_door_pending_hits_{0};
  std::mutex elevator_door_mutex_;
};

}  // namespace wall_plane_filter

#endif
