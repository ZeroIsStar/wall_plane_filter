#ifndef WALL_PLANE_FILTER__LIDAR_TAIL_FILTER_HPP_
#define WALL_PLANE_FILTER__LIDAR_TAIL_FILTER_HPP_

#include "wall_plane_filter/types.hpp"

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace wall_plane_filter
{

/// Lidar-frame Z band cut, then per-ring wall-edge bow removal.
/// z_axis is the lidar +Z axis expressed in the same frame as the cloud.
void runLidarTailPipeline(
  const pcl::PointCloud<pcl::PointXYZ> & input,
  const Eigen::Vector3d & lidar_origin,
  const Eigen::Vector3d & lidar_z_axis,
  const Params & params,
  pcl::PointCloud<pcl::PointXYZ> & filtered,
  pcl::PointCloud<pcl::PointXYZ> & deviate);

}  // namespace wall_plane_filter

#endif
