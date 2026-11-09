#pragma once

#include <cstdint>
#include <vector>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace wall_plane_filter
{
struct HoughConfig
{
  double resolution{0.05};
  double half_extent{6.0};
  int max_grid_side{512};
  bool follow_sensor{true};
  double origin_x{-6.0}, origin_y{-6.0};
  int min_cell_points{2};
  double min_cell_height{0.05};
  // Height band is applied in base_link BEFORE projection/TF into map.
  double z_min{0.15};
  double z_max{1.50};
  double theta_deg{1.0};
  int votes{10};
  double min_line_length{0.5};
  double hough_max_gap{0.15};
  double merge_angle_deg{5.0};
  double merge_distance{0.06};
  double merge_gap{0.15};
  double parallel_angle_deg{10.0};
  double intersection_extension{0.20};
  double support_band{0.08};
  double min_support_ratio{0.70};
  double max_unsupported_gap{0.18};
  double min_continuous_support{0.15};
  int min_support_points{10};
  int max_lines{80};
  bool refine{true};
  bool debug{false};
};

struct WallLine
{
  Eigen::Vector2d a{0, 0}, b{0, 0};
  // Unit normal: normal.dot(xy) = rho. Endpoints bound the supported interval.
  Eigen::Vector2d normal{0, 1};
  double rho{0}, z_min{0}, z_max{0}, sigma{0}, support_ratio{0};
  size_t support_points{0};
};

struct WallProjection
{
  Eigen::Vector2d origin{0, 0};
  double resolution{0.05};
  int width{0}, height{0};
  // Row y increases with world Y; exactly the OccupancyGrid convention.
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
};

// Throws invalid_argument for unsafe/inconsistent settings; never enlarges the
// grid beyond max_grid_side. Input points already passed the sensor height ROI.
void validateHoughConfig(const HoughConfig & c);
HoughResult extractHoughWalls(
  const pcl::PointCloud<pcl::PointXYZ> & cloud, const Eigen::Vector2d & center,
  const HoughConfig & config);
}  // namespace wall_plane_filter
