#include "wall_plane_filter/lidar_tail_filter.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace wall_plane_filter
{
namespace
{

using CloudT = pcl::PointCloud<pcl::PointXYZ>;

void cutLidarZBand(
  const CloudT & input, const Eigen::Vector3d & origin, const Eigen::Vector3d & z_axis,
  const Params & params, std::vector<char> & drop)
{
  if (!params.lidar_z_cut_enable) {
    return;
  }
  const double down = params.lidar_z_cut_down;
  const double up = params.lidar_z_cut_up;
  const double range_max = params.lidar_z_cut_range_max;
  for (size_t i = 0; i < input.size(); ++i) {
    const auto & p = input.points[i];
    const Eigen::Vector3d d(p.x - origin.x(), p.y - origin.y(), p.z - origin.z());
    const double z_l = d.dot(z_axis);
    if (z_l < down || z_l > up) {
      continue;
    }
    // Distance from the lidar z axis. Far wall points at the sensor height stay.
    const double radial = (d - z_l * z_axis).norm();
    if (!(range_max > 0.0) || radial < range_max) {
      drop[i] = 1;
    }
  }
}

void filterWallEdge(
  const CloudT & input, const Eigen::Vector3d & origin, const Eigen::Vector3d & z_axis,
  const Params & params, std::vector<char> & drop)
{
  if (!params.wall_edge_enable || input.empty()) {
    return;
  }
  Eigen::Vector3d z = z_axis;
  if (z.norm() < 1e-6) {
    z = Eigen::Vector3d::UnitZ();
  }
  z.normalize();
  Eigen::Vector3d x_axis = z.cross(Eigen::Vector3d::UnitX());
  if (x_axis.norm() < 0.1) {
    x_axis = z.cross(Eigen::Vector3d::UnitY());
  }
  x_axis.normalize();
  const Eigen::Vector3d y_axis = z.cross(x_axis).normalized();

  const double el_res = std::max(0.2, params.ap_el_res_deg) * M_PI / 180.0;
  const int nel = std::max(8, static_cast<int>(std::ceil(M_PI / el_res)));
  const double half = std::max(2.0, params.ring_bow_half_deg) * M_PI / 180.0;

  struct Sample
  {
    size_t index;
    double az;
    double r;
  };
  std::vector<std::vector<Sample>> rings(static_cast<size_t>(nel));

  for (size_t i = 0; i < input.size(); ++i) {
    if (drop[i]) {
      continue;
    }
    const auto & p = input.points[i];
    const Eigen::Vector3d d(p.x - origin.x(), p.y - origin.y(), p.z - origin.z());
    const double zl = d.dot(z);
    const double xl = d.dot(x_axis);
    const double yl = d.dot(y_axis);
    const double r = d.norm();
    if (r < params.range_min || r > params.range_max) {
      continue;
    }
    double e = std::atan2(zl, std::hypot(xl, yl)) + 0.5 * M_PI;
    if (e < 0.0) {
      e = 0.0;
    }
    if (e >= M_PI) {
      e = M_PI - 1e-9;
    }
    const int ie = std::min(nel - 1, static_cast<int>(e / M_PI * nel));
    rings[static_cast<size_t>(ie)].push_back(Sample{i, std::atan2(yl, xl), r});
  }

  const double edge_min = params.edge_min;
  const double edge_max = params.edge_max;
  for (auto & ring : rings) {
    if (ring.size() < 8) {
      continue;
    }
    std::sort(ring.begin(), ring.end(), [](const Sample & a, const Sample & b) {
      return a.az < b.az;
    });
    const int n = static_cast<int>(ring.size());
    std::vector<double> trend(static_cast<size_t>(n));
    std::vector<char> off(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
      int j0 = i;
      int j1 = i;
      while (j0 > 0 && ring[i].az - ring[j0 - 1].az <= half) {--j0;}
      while (j1 + 1 < n && ring[j1 + 1].az - ring[i].az <= half) {++j1;}
      const int count = j1 - j0 + 1;
      if (count < 5) {
        trend[static_cast<size_t>(i)] = ring[i].r;
        continue;
      }
      // Range vs azimuth, with azimuth shifted so the fit is well conditioned.
      // A slanted wall is nearly linear and stays; a narrow bow is the residual.
      double sum_w = 0, sum_a = 0, sum_r = 0, sum_aa = 0, sum_ar = 0;
      const double az0 = ring[static_cast<size_t>(i)].az;
      for (int j = j0; j <= j1; ++j) {
        const double az = ring[static_cast<size_t>(j)].az - az0;
        const double r = ring[static_cast<size_t>(j)].r;
        sum_w += 1.0;
        sum_a += az;
        sum_r += r;
        sum_aa += az * az;
        sum_ar += az * r;
      }
      const double det = sum_w * sum_aa - sum_a * sum_a;
      double fitted = ring[i].r;
      if (std::abs(det) > 1e-8) {
        const double slope = (sum_w * sum_ar - sum_a * sum_r) / det;
        const double intercept = (sum_r - slope * sum_a) / sum_w;
        fitted = intercept;
      }
      trend[static_cast<size_t>(i)] = fitted;
      const double toward = fitted - ring[i].r;
      if (toward >= edge_min && toward <= edge_max) {
        off[static_cast<size_t>(i)] = 1;
      }
    }
    int i = 0;
    // Burrs are much narrower than the trend window. A long residual from a
    // slanted wall is kept.
    const double max_bow_width = 0.5 * half;
    while (i < n) {
      if (!off[static_cast<size_t>(i)]) {
        ++i;
        continue;
      }
      int j = i;
      double peak = 0.0;
      while (j < n && off[static_cast<size_t>(j)]) {
        peak = std::max(peak, trend[static_cast<size_t>(j)] - ring[static_cast<size_t>(j)].r);
        ++j;
      }
      const bool both_shoulders = i > 0 && j < n;
      const double width = ring[static_cast<size_t>(j - 1)].az - ring[static_cast<size_t>(i)].az;
      if (both_shoulders && peak <= params.ap_keep_depth && width < max_bow_width) {
        for (int k = i; k < j; ++k) {
          drop[ring[static_cast<size_t>(k)].index] = 1;
        }
      }
      i = j;
    }
  }
}

}  // namespace

void runLidarTailPipeline(
  const CloudT & input, const Eigen::Vector3d & lidar_origin, const Eigen::Vector3d & lidar_z_axis,
  const Params & params, CloudT & filtered, CloudT & deviate)
{
  filtered.clear();
  deviate.clear();
  std::vector<char> drop(input.size(), 0);
  cutLidarZBand(input, lidar_origin, lidar_z_axis, params, drop);
  filterWallEdge(input, lidar_origin, lidar_z_axis, params, drop);
  filtered.reserve(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    if (drop[i]) {
      deviate.push_back(input.points[i]);
    } else {
      filtered.push_back(input.points[i]);
    }
  }
  filtered.width = static_cast<uint32_t>(filtered.size());
  filtered.height = 1;
  filtered.is_dense = false;
  deviate.width = static_cast<uint32_t>(deviate.size());
  deviate.height = 1;
  deviate.is_dense = false;
}

}  // namespace wall_plane_filter
