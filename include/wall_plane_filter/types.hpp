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
  HoughConfig depth_hough;
  HoughConfig lidar_hough;
  WallTrackConfig wall_track;
  double wall_line_width{0.04};
  double wall_display_z{0.10};
  double wall_marker_lifetime{2.0};
  double wall_marker_publish_hz{20.0};

  std::string target_frame{"base_link"};
  double tf_timeout{0.25};

  // Lidar: ROI + voxel → extract → WallTracker+KF
  std::string cloud_topic{"/vanjee/lidar"};
  bool lidar_wall_enable{true};
  bool publish_cloud_filtered{false};
  double z_min{0.05};
  double z_max{1.80};
  double range_min{0.20};
  double range_max{3.0};              // lidar ROI horizontal max (m)
  double voxel_leaf{0.05};
  double lidar_wall_range_max{3.0};   // lidar wall-extract range (m)

  // Corner lock + support extent clamp (shared by depth / lidar publish paths).
  bool corner_snap_enable{true};
  double corner_snap_dist{0.0};       // ≤0 → use hough.intersection_extension
  bool wall_clamp_to_support{true};
  double wall_extent_margin_m{0.01};  // slack beyond support t/z extremes (m)

  bool depth_clean_enable{true};
  std::vector<std::string> depth_topics{
    "/camera1/depth/points",
    "/camera2/depth/points",
    "/camera3/depth/points",
    "/camera4/depth/points"};
  std::vector<std::string> depth_wall_cameras{"camera1", "camera2", "camera3", "camera4"};

  // Preprocess: crop → isotropic VoxelGrid → (optional ROR) → SOR → bilateral.
  double depth_z_min{0.05};
  double depth_z_max{1.70};
  double depth_range_min{0.10};
  double depth_range_max{3.0};
  double depth_voxel_xy{0.02};  // isotropic leaf (m): leaf×leaf×leaf
  bool depth_ror_enable{true};
  int depth_ror_min_points{200};       // skip ROR when fewer points
  double depth_ror_radius{0.05};       // RadiusOutlierRemoval search radius (m)
  int depth_ror_min_neighbors{4};      // min neighbors in radius
  bool depth_sor_enable{true};
  int depth_sor_mean_k{20};
  double depth_sor_stddev_mul{1.0};
  bool depth_bilateral_enable{true};
  double depth_bilateral_sigma_s{0.04};
  double depth_bilateral_sigma_r{0.03};

  // Publish cull onto vertical walls (after track).
  double depth_wall_band_m{0.05};
  double depth_wall_band_outer_m{0.12};
  bool depth_wall_project{true};
  double depth_wall_noise_k{0.005};
  double depth_wall_noise_max{0.06};
  double depth_wall_range_max{3.0};
  double depth_wall_ground_margin{0.08};
  double depth_ground_ref_z{0.0};
  int wall_fuse_min_support_points{10};
  double depth_merge_max_age{0.5};
  double depth_wall_tf_report_sec{5.0};

  std::string depth_wall_map_frame{"map"};
  std::string depth_wall_output_frame{"map"};
};

}  // namespace wall_plane_filter

#endif  // WALL_PLANE_FILTER__TYPES_HPP_
