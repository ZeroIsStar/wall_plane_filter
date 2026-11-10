#ifndef WALL_PLANE_FILTER__TYPES_HPP_
#define WALL_PLANE_FILTER__TYPES_HPP_

#include <string>
#include <vector>
#include "wall_plane_filter/wall_hough.hpp"
#include "wall_plane_filter/wall_track.hpp"

namespace wall_plane_filter
{

struct Params
{
  HoughConfig depth_hough, lidar_hough;
  // Temporal association / coasting shared by depth and lidar trackers.
  WallTrackConfig wall_track;
  double wall_line_width{0.04};
  double wall_display_z{0.10};
  // Keep markers longer than lose_sec so coasted walls do not blink away.
  double wall_marker_lifetime{2.0};
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
  // Wall-band cull uses |n·xy - rho| (plane-normal distance) on finite segments.
  // Effective keep = max(band, min(noise_max, noise_k·r²)) so far walls flatten too.
  // |d| <= keep: keep (optionally project onto the plane).
  // keep < |d| <= outer: smear → deviate.
  // |d| > outer (or off-segment): keep as obstacle / free space.
  double depth_wall_band_m{0.07};
  double depth_wall_band_outer_m{0.20};
  // Project in-band wall points onto the vertical plane so the cloud is coplanar.
  bool depth_wall_project{true};
  // PCA flatten when no tracked wall yet (skipped once walls exist — use wall project).
  bool depth_local_flatten_enable{true};
  double depth_local_flatten_radius{0.12};
  int depth_local_flatten_min_neighbors{10};
  // λ0/(λ0+λ1+λ2); lower = flatter. Above this, leave the point alone.
  double depth_local_flatten_max_curvature{0.08};
  // Cap PCA points so flatten cannot starve depth Hough callbacks.
  int depth_local_flatten_max_points{2000};
  // Organized depth image: drop a pixel when enough neighbors disagree in range.
  bool depth_fly_enable{true};
  double depth_fly_abs_m{0.04};
  double depth_fly_rel{0.03};
  // Neighbors over threshold required to mark flying (1..8). Default 2.
  int depth_fly_min_neighbors{2};
  // Median of range in a (2*half+1)^2 window, then keep every Nth pixel.
  bool depth_median_enable{true};
  int depth_median_half{2};
  int depth_median_min{8};
  // Skip median scale when neighborhood MAD exceeds this (m); keeps edges sharp.
  double depth_median_edge_mad{0.03};
  int depth_decimate_stride{2};
  // Optional local-flatness gate before wall projection (default off = old behavior).
  bool wall_project_local_check{false};
  double wall_project_radius{0.15};
  double wall_project_normal_deg{25.0};
  double wall_project_max_curvature{0.08};
  int wall_project_min_neighbors{6};
  // Cross-camera wall fuse: drop lines with fewer support points.
  int wall_fuse_min_support_points{10};
  // true: cull with fused global walls (legacy). false: prefer per-camera tracks.
  bool depth_cull_use_global_walls{true};
  // Radius filter runs after the voxel grid. Isolated voxels are real specks.
  bool depth_radius_enable{true};
  double depth_radius_search{0.08};
  int depth_radius_min_neighbors{4};
  double depth_wall_range_max{3.0};
  double depth_wall_fuse_hz{10.0};
  std::vector<std::string> depth_wall_cameras{"camera1", "camera2", "camera3", "camera4"};
  double depth_merge_max_age{0.5};
  double depth_wall_noise_k{0.005};
  double depth_wall_noise_max{0.08};
  double depth_wall_ground_margin{0.08};
  double depth_ground_ref_z{0.0};
  double depth_wall_tf_report_sec{5.0};


  std::string depth_wall_map_frame{"map"};
  std::string depth_wall_output_frame{"map"};

  // Elevator door open/closed in target_frame (usually base_link) forward ROI.
  bool elevator_door_enable{true};
  double elevator_door_x_min{0.40};
  double elevator_door_x_max{2.50};
  double elevator_door_half_width{0.55};
  double elevator_door_z_min{0.20};
  double elevator_door_z_max{1.40};
  // Columns farther than near_x + open_margin count as gap (open).
  double elevator_door_open_margin{0.35};
  // open_bins/valid_bins above this → OPEN, else CLOSED (when enough points).
  double elevator_door_open_ratio{0.45};
  int elevator_door_bins{11};
  int elevator_door_min_points_bin{3};
  int elevator_door_min_valid_bins{4};
  // Consecutive agreeing frames before publishing a state change.
  int elevator_door_vote_frames{4};
};

}  // namespace wall_plane_filter

#endif  // WALL_PLANE_FILTER__TYPES_HPP_
