#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace wall_plane_filter
{

// extractor: "hough" | "ransac" | "seed_grow"
struct HoughConfig
{
  std::string extractor{"seed_grow"};
  double resolution{0.05};
  double half_extent{3.0};  // XY window half-size; keep ≥ ROI range (3 m)
  int max_grid_side{512};
  bool follow_sensor{true};
  double origin_x{-3.0}, origin_y{-3.0};
  int min_cell_points{2};
  double min_cell_height{0.05};
  double z_min{0.05};
  double z_max{1.70};
  double theta_deg{1.0};
  int votes{10};
  double min_line_length{0.40};
  double hough_max_gap{0.15};
  double merge_angle_deg{6.0};
  double merge_distance{0.08};
  double merge_gap{0.25};
  double parallel_angle_deg{10.0};
  double intersection_extension{0.50};
  double support_band{0.12};
  double min_support_ratio{0.65};
  double max_unsupported_gap{0.20};
  double min_continuous_support{0.15};
  int min_support_points{8};
  int max_lines{80};
  bool refine{true};
  bool debug{false};
  // RANSAC
  double ransac_distance{0.0};
  int ransac_min_inliers{20};
  int ransac_max_iterations{150};
  double ransac_downsample_leaf{0.04};
  // seed_grow
  double seed_contour_angle_deg{1.0};
  int seed_min_points{6};
  double seed_max_distance{0.0};
  double grow_distance{0.0};
  int grow_max_gap_bins{2};
  double grow_max_range_jump{0.35};
};

struct WallLine
{
  Eigen::Vector2d a{0, 0}, b{0, 0};
  Eigen::Vector2d normal{0, 1};
  double rho{0}, z_min{0}, z_max{0}, sigma{0}, support_ratio{0};
  size_t support_points{0};
};

struct WallProjection
{
  Eigen::Vector2d origin{0, 0};
  double resolution{0.05};
  int width{0}, height{0};
  std::vector<uint32_t> counts;
  std::vector<uint8_t> binary;
};

struct HoughResult
{
  WallProjection grid;
  std::vector<WallLine> hough_lines, candidates, lines;
  std::vector<Eigen::Vector2d> intersections, corners;
  size_t input_points{0}, projected_points{0}, occupied_cells{0};
  double elapsed_ms{0};
  std::string extractor{"seed_grow"};
};

void validateHoughConfig(const HoughConfig & c);
HoughResult extractHoughWalls(
  const pcl::PointCloud<pcl::PointXYZ> & cloud, const Eigen::Vector2d & center,
  const HoughConfig & config);

/** Snap nearby non-parallel wall endpoints onto their intersection (L-corner lock). */
void snapWallCorners(
  std::vector<WallLine> & lines, double parallel_angle_deg, double snap_dist,
  double min_line_length);

/**
 * Hard-clamp each wall's a–b and z to support points in `cloud` within support_band
 * of the segment (plus optional margin). Drops walls that become too short.
 */
void clampWallsToSupportCloud(
  std::vector<WallLine> & lines, const pcl::PointCloud<pcl::PointXYZ> & cloud,
  double support_band, double margin, double min_line_length);

}  // namespace wall_plane_filter
