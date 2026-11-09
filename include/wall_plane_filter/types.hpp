#ifndef WALL_PLANE_FILTER__TYPES_HPP_
#define WALL_PLANE_FILTER__TYPES_HPP_

#include <string>
#include <vector>
#include "wall_plane_filter/wall_hough.hpp"

namespace wall_plane_filter
{

struct Params
{
  HoughConfig depth_hough, lidar_hough;
  double wall_line_width{0.04};
  double wall_display_z{0.10};
  double wall_marker_lifetime{0.75};
  std::string cloud_topic{"/vanjee/lidar"};
  std::string target_frame{"base_link"};
  // Empty → use the PointCloud2 header.frame_id as the lidar frame.
  std::string lidar_frame{""};

  double z_min{0.05};
  double z_max{1.80};
  double range_min{0.20};
  double range_max{12.0};
  double voxel_leaf{0.02};
  double process_hz{10.0};
  double tf_timeout{0.25};
  bool lidar_sub_reliable{false};

  bool lidar_z_cut_enable{true};
  double lidar_z_cut_down{-0.15};
  double lidar_z_cut_up{0.15};
  // Drop the lidar-z band only inside this distance from the lidar axis (m).
  double lidar_z_cut_range_max{1.5};
  bool lidar_wall_enable{true};
  double lidar_wall_range_max{3.0};
  bool wall_edge_enable{true};
  double edge_min{0.04};
  double edge_max{0.45};
  double ring_bow_half_deg{12.0};
  double ap_el_res_deg{1.0};
  double ap_keep_depth{0.40};

  bool depth_clean_enable{true};
  std::vector<std::string> depth_topics{
    "/camera1/depth/points",
    "/camera2/depth/points",
    "/camera3/depth/points",
    "/camera4/depth/points"};
  // Reliable subscriptions let DDS repair lost fragments of large clouds.
  bool depth_sub_reliable{true};
  double depth_z_min{-0.05};
  double depth_z_max{1.50};
  double depth_range_min{0.10};
  double depth_range_max{4.0};
  double depth_voxel_leaf{0.05};
  // Cloud published for viewing. Finer than the evidence voxel so a 2 cm
  // wall band can actually hold the points.
  double depth_publish_voxel{0.02};
  // Drop points whose horizontal distance to a published wall is in
  // (band, band_outer]. Closer points stay on the wall. Farther points are
  // obstacles and stay.
  double depth_wall_band_m{0.02};
  double depth_wall_band_outer_m{0.15};
  // Organized depth image: drop a pixel when its range disagrees with any
  // 4-neighbor by more than max(abs, rel * range). Cuts flying edge pixels.
  bool depth_fly_enable{true};
  double depth_fly_abs_m{0.04};
  double depth_fly_rel{0.03};
  // 3x3 median of range along the camera ray, then keep every Nth pixel.
  bool depth_median_enable{true};
  int depth_median_min{5};
  int depth_decimate_stride{2};
  // Radius filter runs after the voxel grid. Isolated voxels are real specks.
  bool depth_radius_enable{true};
  double depth_radius_search{0.08};
  int depth_radius_min_neighbors{4};
  double depth_wall_range_max{3.0};
  double depth_wall_fuse_hz{10.0};
  std::vector<std::string> depth_wall_cameras{"camera1", "camera2", "camera3", "camera4"};
  double depth_merge_max_age{0.5};
  double depth_wall_noise_k{0.005};
  double depth_wall_noise_max{0.045};
  double depth_wall_ground_margin{0.08};
  double depth_ground_ref_z{0.0};
  double depth_wall_tf_report_sec{5.0};


  std::string depth_wall_map_frame{"map"};
  std::string depth_wall_output_frame{"map"};

};

}  // namespace wall_plane_filter

#endif  // WALL_PLANE_FILTER__TYPES_HPP_
