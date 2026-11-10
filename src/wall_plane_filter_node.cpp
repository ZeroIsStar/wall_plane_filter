#include "wall_plane_filter/wall_plane_filter_node.hpp"
#include "wall_plane_filter/lidar_tail_filter.hpp"
#include "wall_plane_filter/wall_hough.hpp"
#include "wall_plane_filter/wall_track.hpp"

#include <tf2_eigen/tf2_eigen.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace wall_plane_filter
{
namespace
{

using CloudT = pcl::PointCloud<pcl::PointXYZ>;

void finishCloud(CloudT & cloud)
{
  cloud.width = static_cast<uint32_t>(cloud.size());
  cloud.height = 1;
  cloud.is_dense = false;
}

float cameraRange(const pcl::PointXYZ & p)
{
  if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const float r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
  return r > 1e-4f ? r : std::numeric_limits<float>::quiet_NaN();
}

// Flying pixels (8-neigh), optional median with edge MAD gate, then stride decimation.
// Organized clouds only; output stays on the camera ray. fly gets rejected edges.
void denoiseDepthImage(const CloudT & in, const Params & params, CloudT & kept, CloudT & fly)
{
  kept.clear();
  fly.clear();
  const bool organized =
    in.height > 1 && in.width > 1 && in.points.size() == static_cast<size_t>(in.width) * in.height;
  if (!organized) {
    kept.reserve(in.size());
    for (const auto & p : in.points) {
      if (std::isfinite(cameraRange(p))) {
        kept.push_back(p);
      }
    }
    finishCloud(kept);
    finishCloud(fly);
    return;
  }

  const int w = static_cast<int>(in.width);
  const int h = static_cast<int>(in.height);
  const size_t n = in.points.size();
  std::vector<float> range(n);
  for (size_t i = 0; i < n; ++i) {
    range[i] = cameraRange(in.points[i]);
  }

  std::vector<uint8_t> flying(n, 0);
  if (params.depth_fly_enable) {
    const float abs_m = static_cast<float>(std::max(0.0, params.depth_fly_abs_m));
    const float rel = static_cast<float>(std::max(0.0, params.depth_fly_rel));
    const int need_bad = std::clamp(params.depth_fly_min_neighbors, 1, 8);
    const int du[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dv[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    for (int v = 0; v < h; ++v) {
      for (int u = 0; u < w; ++u) {
        const size_t i = static_cast<size_t>(v * w + u);
        const float r = range[i];
        if (!std::isfinite(r)) {
          continue;
        }
        const float lim = std::max(abs_m, rel * r);
        int bad = 0;
        for (int k = 0; k < 8; ++k) {
          const int uu = u + du[k];
          const int vv = v + dv[k];
          if (uu < 0 || vv < 0 || uu >= w || vv >= h) {
            continue;
          }
          const float rn = range[static_cast<size_t>(vv * w + uu)];
          if (!std::isfinite(rn)) {
            continue;
          }
          if (std::fabs(r - rn) > lim) {
            ++bad;
            if (bad >= need_bad) {
              flying[i] = 1;
              break;
            }
          }
        }
      }
    }
  }

  fly.reserve(n / 8);
  for (size_t i = 0; i < n; ++i) {
    if (flying[i]) {
      fly.push_back(in.points[i]);
    }
  }

  std::vector<float> med = range;
  if (params.depth_median_enable) {
    const int half = std::clamp(params.depth_median_half, 1, 3);
    const int need = std::max(1, params.depth_median_min);
    const float edge_mad = static_cast<float>(std::max(0.0, params.depth_median_edge_mad));
    std::vector<float> window(static_cast<size_t>((2 * half + 1) * (2 * half + 1)));
    for (int v = 0; v < h; ++v) {
      for (int u = 0; u < w; ++u) {
        const size_t i = static_cast<size_t>(v * w + u);
        if (flying[i] || !std::isfinite(range[i])) {
          med[i] = std::numeric_limits<float>::quiet_NaN();
          continue;
        }
        int count = 0;
        for (int dv = -half; dv <= half; ++dv) {
          for (int du = -half; du <= half; ++du) {
            const int uu = u + du;
            const int vv = v + dv;
            if (uu < 0 || vv < 0 || uu >= w || vv >= h) {
              continue;
            }
            const size_t j = static_cast<size_t>(vv * w + uu);
            if (flying[j] || !std::isfinite(range[j])) {
              continue;
            }
            window[static_cast<size_t>(count++)] = range[j];
          }
        }
        if (count < need) {
          med[i] = std::numeric_limits<float>::quiet_NaN();
          continue;
        }
        std::nth_element(window.begin(), window.begin() + count / 2, window.begin() + count);
        const float median = window[static_cast<size_t>(count / 2)];
        // MAD: skip median scale on strong edges (zipper).
        if (edge_mad > 0.0f) {
          for (int c = 0; c < count; ++c) {
            window[static_cast<size_t>(c)] = std::fabs(window[static_cast<size_t>(c)] - median);
          }
          std::nth_element(window.begin(), window.begin() + count / 2, window.begin() + count);
          const float mad = window[static_cast<size_t>(count / 2)];
          if (mad > edge_mad) {
            med[i] = range[i];
            continue;
          }
        }
        med[i] = median;
      }
    }
  }

  const int stride = std::max(1, params.depth_decimate_stride);
  const int v0 = stride / 2;
  const int u0 = stride / 2;
  kept.reserve(n / static_cast<size_t>(stride * stride));
  for (int v = v0; v < h; v += stride) {
    for (int u = u0; u < w; u += stride) {
      const size_t i = static_cast<size_t>(v * w + u);
      const float r = range[i];
      const float m = med[i];
      if (flying[i] || !std::isfinite(r) || !std::isfinite(m)) {
        continue;
      }
      const float scale = m / r;
      const auto & p = in.points[i];
      kept.push_back(pcl::PointXYZ(p.x * scale, p.y * scale, p.z * scale));
    }
  }
  finishCloud(kept);
  finishCloud(fly);
}

void voxelLeaf(const CloudT & in, double leaf, CloudT & out)
{
  if (leaf <= 1e-4 || in.empty()) {
    out = in;
    finishCloud(out);
    return;
  }
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(in.makeShared());
  const float side = static_cast<float>(leaf);
  vg.setLeafSize(side, side, side);
  vg.filter(out);
  finishCloud(out);
}

void keepInRoi(
  const CloudT & in, const Eigen::Isometry3d & T_bc, const Params & params, CloudT & out)
{
  out.clear();
  out.reserve(in.size());
  for (const auto & p : in.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      continue;
    }
    const Eigen::Vector3d q = T_bc * Eigen::Vector3d(p.x, p.y, p.z);
    if (q.z() < params.depth_z_min || q.z() > params.depth_z_max) {
      continue;
    }
    const double r = std::hypot(q.x(), q.y());
    if (r < params.depth_range_min || r > params.depth_range_max) {
      continue;
    }
    out.push_back(pcl::PointXYZ(
      static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z())));
  }
  finishCloud(out);
}

void transformInto(const CloudT & in, const Eigen::Isometry3d & T, CloudT & out)
{
  out.clear();
  out.reserve(in.size());
  for (const auto & p : in.points) {
    const Eigen::Vector3d q = T * Eigen::Vector3d(p.x, p.y, p.z);
    out.push_back(pcl::PointXYZ(
      static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z())));
  }
  finishCloud(out);
}

geometry_msgs::msg::Point pointMsg(const Eigen::Vector3d & p)
{
  geometry_msgs::msg::Point g;
  g.x = p.x();
  g.y = p.y();
  g.z = p.z();
  return g;
}

WallLine transformLine(const WallLine & line, const Eigen::Isometry3d & tf)
{
  const double z0 = std::isfinite(line.z_min) ? line.z_min : 0.0;
  const double z1 = std::isfinite(line.z_max) ? line.z_max : z0;
  const Eigen::Vector3d a_lo = tf * Eigen::Vector3d(line.a.x(), line.a.y(), z0);
  const Eigen::Vector3d a_hi = tf * Eigen::Vector3d(line.a.x(), line.a.y(), z1);
  const Eigen::Vector3d b_lo = tf * Eigen::Vector3d(line.b.x(), line.b.y(), z0);
  const Eigen::Vector3d b_hi = tf * Eigen::Vector3d(line.b.x(), line.b.y(), z1);
  WallLine out = line;
  out.a = a_lo.head<2>();
  out.b = b_lo.head<2>();
  out.z_min = std::min(std::min(a_lo.z(), a_hi.z()), std::min(b_lo.z(), b_hi.z()));
  out.z_max = std::max(std::max(a_lo.z(), a_hi.z()), std::max(b_lo.z(), b_hi.z()));
  const Eigen::Vector2d d = out.b - out.a;
  if (d.norm() <= 1e-6) {
    return line;
  }
  const Eigen::Vector2d dir = d.normalized();
  out.normal = Eigen::Vector2d(-dir.y(), dir.x());
  if (out.normal.dot(line.normal) < 0.0) {
    out.normal = -out.normal;
  }
  out.rho = out.normal.dot(out.a);
  return out;
}

std::vector<WallLine> transformLines(const std::vector<WallLine> & lines, const Eigen::Isometry3d & tf)
{
  std::vector<WallLine> out;
  out.reserve(lines.size());
  for (const auto & line : lines) {
    out.push_back(transformLine(line, tf));
  }
  return out;
}

std::vector<Eigen::Vector2d> transformPoints(
  const std::vector<Eigen::Vector2d> & points, const Eigen::Isometry3d & tf)
{
  std::vector<Eigen::Vector2d> out;
  out.reserve(points.size());
  for (const auto & p : points) {
    out.push_back((tf * Eigen::Vector3d(p.x(), p.y(), 0.0)).head<2>());
  }
  return out;
}

// Nearest finite wall by plane-normal distance. Endpoints get a small along-track
// margin so corners are not over-aggressively culled.
bool nearestWallNormal(
  const Eigen::Vector2d & xy, const std::vector<WallLine> & walls, double end_margin,
  double & abs_d, double & signed_d, Eigen::Vector2d & normal_out)
{
  abs_d = std::numeric_limits<double>::infinity();
  signed_d = 0.0;
  normal_out = Eigen::Vector2d::Zero();
  bool found = false;
  for (const auto & wall : walls) {
    Eigen::Vector2d n = wall.normal;
    const double nn = n.norm();
    if (nn < 1e-9) {
      continue;
    }
    n /= nn;
    const double rho = n.dot(wall.a);
    const double sd = n.dot(xy) - rho;
    const double ad = std::abs(sd);

    Eigen::Vector2d dir = wall.b - wall.a;
    const double len = dir.norm();
    if (len < 1e-9) {
      continue;
    }
    dir /= len;
    const double t = dir.dot(xy - wall.a);
    // Off the finite segment: fall back to endpoint distance (no plane project).
    if (t < -end_margin || t > len + end_margin) {
      const double d_end = std::min((xy - wall.a).norm(), (xy - wall.b).norm());
      if (d_end < abs_d) {
        abs_d = d_end;
        signed_d = sd;
        normal_out = Eigen::Vector2d::Zero();  // mark: not on-segment
        found = true;
      }
      continue;
    }
    if (ad < abs_d) {
      abs_d = ad;
      signed_d = sd;
      normal_out = n;
      found = true;
    }
  }
  return found;
}

struct LocalPlaneFeat
{
  bool valid{false};
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
  double curvature{1.0};
};

bool computeLocalPlaneFeat(
  const CloudT & cloud, pcl::KdTreeFLANN<pcl::PointXYZ> & tree, size_t idx,
  double radius, int min_neighbors, LocalPlaneFeat & out)
{
  out.valid = false;
  std::vector<int> ids;
  std::vector<float> dist2;
  const int need = std::max(3, min_neighbors);
  if (tree.radiusSearch(cloud.points[idx], radius, ids, dist2) < need) {
    return false;
  }
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (int id : ids) {
    const auto & q = cloud.points[static_cast<size_t>(id)];
    mean += Eigen::Vector3d(q.x, q.y, q.z);
  }
  mean /= static_cast<double>(ids.size());
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (int id : ids) {
    const auto & q = cloud.points[static_cast<size_t>(id)];
    const Eigen::Vector3d d = Eigen::Vector3d(q.x, q.y, q.z) - mean;
    cov += d * d.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
  if (eig.info() != Eigen::Success) {
    return false;
  }
  const Eigen::Vector3d evals = eig.eigenvalues();
  const double sum = evals.sum();
  if (!(sum > 1e-12)) {
    return false;
  }
  out.curvature = evals(0) / sum;
  out.normal = eig.eigenvectors().col(0);
  out.valid = true;
  return true;
}

// Cull smear around tracked walls and project inliers onto the plane.
// keep is expanded per-point with k·r² noise (r = distance to sensor_xy).
// Optional local-flatness gate avoids projecting near-wall obstacles.
size_t cullWallBand(
  CloudT & cloud, CloudT & deviate, const std::vector<WallLine> & walls,
  double keep, double outer, bool project,
  const Eigen::Vector2d & robot_xy, double noise_k, double noise_max,
  size_t * projected_out = nullptr,
  const Eigen::Vector2d * sensor_xy = nullptr,
  const Params * local_params = nullptr)
{
  if (projected_out) {
    *projected_out = 0;
  }
  if (cloud.empty() || walls.empty() || !(keep > 0.0) || !(outer > keep)) {
    return 0;
  }
  const Eigen::Vector2d noise_origin = sensor_xy ? *sensor_xy : robot_xy;
  const bool local_check = local_params && local_params->wall_project_local_check && project;
  const double end_margin = std::max({keep, outer * 0.5, 0.12});

  pcl::KdTreeFLANN<pcl::PointXYZ> tree;
  std::vector<char> candidate(cloud.size(), 0);
  size_t candidate_n = 0;
  if (local_check) {
    // Only points near a wall band need PCA features.
    for (size_t i = 0; i < cloud.size(); ++i) {
      const Eigen::Vector2d xy(cloud.points[i].x, cloud.points[i].y);
      double abs_d = 0.0;
      double signed_d = 0.0;
      Eigen::Vector2d n = Eigen::Vector2d::Zero();
      if (!nearestWallNormal(xy, walls, end_margin, abs_d, signed_d, n)) {
        continue;
      }
      if (!(n.squaredNorm() > 0.5)) {
        continue;
      }
      const double range = (xy - noise_origin).norm();
      const double noise = std::min(noise_max, noise_k * range * range);
      const double keep_pt = std::max(keep, noise);
      const double outer_pt = std::max(outer, keep_pt + 0.02);
      if (abs_d <= outer_pt) {
        candidate[i] = 1;
        ++candidate_n;
      }
    }
    if (candidate_n > 0) {
      tree.setInputCloud(cloud.makeShared());
    }
  }

  CloudT kept;
  kept.reserve(cloud.size());
  size_t dropped = 0;
  size_t projected = 0;
  const double cos_n =
    local_check ? std::cos(local_params->wall_project_normal_deg * M_PI / 180.0) : 0.0;
  for (size_t i = 0; i < cloud.size(); ++i) {
    const auto & p = cloud.points[i];
    const Eigen::Vector2d xy(p.x, p.y);
    double abs_d = 0.0;
    double signed_d = 0.0;
    Eigen::Vector2d n = Eigen::Vector2d::Zero();
    if (!nearestWallNormal(xy, walls, end_margin, abs_d, signed_d, n)) {
      kept.push_back(p);
      continue;
    }
    const bool on_segment = n.squaredNorm() > 0.5;
    const double range = (xy - noise_origin).norm();
    const double noise = std::min(noise_max, noise_k * range * range);
    const double keep_pt = std::max(keep, noise);
    const double outer_pt = std::max(outer, keep_pt + 0.02);
    if (on_segment && abs_d > keep_pt && abs_d <= outer_pt) {
      deviate.push_back(p);
      ++dropped;
      continue;
    }
    if (on_segment && abs_d <= keep_pt && project) {
      bool do_project = true;
      if (local_check && candidate[i]) {
        LocalPlaneFeat feat;
        if (!computeLocalPlaneFeat(
            cloud, tree, i, local_params->wall_project_radius,
            local_params->wall_project_min_neighbors, feat) ||
          feat.curvature > local_params->wall_project_max_curvature)
        {
          do_project = false;
        } else {
          const Eigen::Vector3d wall_n(n.x(), n.y(), 0.0);
          if (std::abs(feat.normal.dot(wall_n)) < cos_n) {
            do_project = false;
          }
        }
      }
      if (do_project) {
        pcl::PointXYZ q = p;
        q.x = static_cast<float>(p.x - signed_d * n.x());
        q.y = static_cast<float>(p.y - signed_d * n.y());
        kept.push_back(q);
        ++projected;
        continue;
      }
      kept.push_back(p);
      continue;
    }
    kept.push_back(p);
  }
  cloud = std::move(kept);
  finishCloud(cloud);
  finishCloud(deviate);
  if (projected_out) {
    *projected_out = projected;
  }
  return dropped;
}

// Project points that sit on locally flat patches onto their neighborhood plane.
// Caps work to max_points via stride sampling to protect callback latency.
size_t flattenLocalPlanes(
  CloudT & cloud, double radius, int min_neighbors, double max_curvature, int max_points = 2000)
{
  if (cloud.size() < static_cast<size_t>(std::max(3, min_neighbors)) ||
    !(radius > 1e-4) || !(max_curvature > 0.0))
  {
    return 0;
  }
  const int cap = std::max(3, max_points);
  const size_t step = std::max<size_t>(1, cloud.size() / static_cast<size_t>(cap));
  pcl::KdTreeFLANN<pcl::PointXYZ> tree;
  auto input = cloud.makeShared();
  tree.setInputCloud(input);
  CloudT out = cloud;
  size_t flattened = 0;
  const int need = std::max(3, min_neighbors);
  for (size_t i = 0; i < cloud.size(); i += step) {
    LocalPlaneFeat feat;
    if (!computeLocalPlaneFeat(cloud, tree, i, radius, need, feat)) {
      continue;
    }
    if (feat.curvature > max_curvature) {
      continue;
    }
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    std::vector<int> ids;
    std::vector<float> dist2;
    if (tree.radiusSearch(cloud.points[i], radius, ids, dist2) < need) {
      continue;
    }
    for (int id : ids) {
      const auto & q = cloud.points[static_cast<size_t>(id)];
      mean += Eigen::Vector3d(q.x, q.y, q.z);
    }
    mean /= static_cast<double>(ids.size());
    const Eigen::Vector3d p(cloud.points[i].x, cloud.points[i].y, cloud.points[i].z);
    const Eigen::Vector3d q = p - feat.normal * feat.normal.dot(p - mean);
    out.points[i].x = static_cast<float>(q.x());
    out.points[i].y = static_cast<float>(q.y());
    out.points[i].z = static_cast<float>(q.z());
    ++flattened;
  }
  cloud.swap(out);
  finishCloud(cloud);
  return flattened;
}

bool tryFuseWallLine(
  WallLine & a, const WallLine & b, double merge_angle_deg, double merge_distance, double merge_gap)
{
  const Eigen::Vector2d da = (a.b - a.a);
  const double la = da.norm();
  if (la < 1e-9) {
    return false;
  }
  const Eigen::Vector2d d = da / la;
  const Eigen::Vector2d db = (b.b - b.a);
  const double lb = db.norm();
  if (lb < 1e-9) {
    return false;
  }
  const Eigen::Vector2d dbn = db / lb;
  if (std::abs(d.dot(dbn)) < std::cos(merge_angle_deg * M_PI / 180.0)) {
    return false;
  }
  if (std::abs(a.normal.dot(b.a) - a.rho) > merge_distance ||
    std::abs(a.normal.dot(b.b) - a.rho) > merge_distance)
  {
    return false;
  }
  const double t0 = d.dot(b.a - a.a);
  const double t1 = d.dot(b.b - a.a);
  const double lo = std::min(t0, t1);
  const double hi = std::max(t0, t1);
  if (lo > la + merge_gap || hi < -merge_gap) {
    return false;
  }
  const double wa = static_cast<double>(std::max<size_t>(1, a.support_points));
  const double wb = static_cast<double>(std::max<size_t>(1, b.support_points));
  const double ratio_a = a.support_ratio;
  const double sigma_a = a.sigma;
  Eigen::Vector2d n = (wa * a.normal + wb * b.normal).normalized();
  if (n.dot(a.normal) < 0.0) {
    n = -n;
  }
  const Eigen::Vector2d a0 = a.a + d * std::min(0.0, lo);
  const Eigen::Vector2d a1 = a.a + d * std::max(la, hi);
  a.a = a0;
  a.b = a1;
  a.normal = n;
  a.rho = n.dot(a.a);
  a.z_min = std::min(a.z_min, b.z_min);
  a.z_max = std::max(a.z_max, b.z_max);
  a.support_points = a.support_points + b.support_points;
  a.support_ratio = (wa * ratio_a + wb * b.support_ratio) / (wa + wb);
  a.sigma = (wa * sigma_a + wb * b.sigma) / (wa + wb);
  return true;
}

std::vector<WallLine> fuseWallLines(
  const std::vector<WallLine> & lines, double merge_angle_deg, double merge_distance,
  double merge_gap, int min_support_points)
{
  std::vector<WallLine> filtered;
  filtered.reserve(lines.size());
  const size_t min_pts = static_cast<size_t>(std::max(1, min_support_points));
  for (const auto & line : lines) {
    if (line.support_points >= min_pts) {
      filtered.push_back(line);
    }
  }
  std::vector<WallLine> merged;
  merged.reserve(filtered.size());
  for (const auto & line : filtered) {
    bool fused = false;
    for (auto & prev : merged) {
      if (tryFuseWallLine(prev, line, merge_angle_deg, merge_distance, merge_gap)) {
        fused = true;
        break;
      }
    }
    if (!fused) {
      merged.push_back(line);
    }
  }
  return merged;
}

}  // namespace

WallPlaneFilterNode::WallPlaneFilterNode(const rclcpp::NodeOptions & options)
: Node("wall_plane_filter", options)
{
  loadParams();

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Dual QoS: this workspace's RViz PointCloud2 displays are Reliable; defaults are BE.
  const auto cloud_rel = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
  const auto cloud_be = rclcpp::SensorDataQoS().keep_last(5);
  filtered_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "/wall_plane_filter/cloud_filtered", cloud_rel);
  filtered_pub_be_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "/wall_plane_filter/cloud_filtered", cloud_be);
  deviate_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "/wall_plane_filter/cloud_deviate", cloud_rel);
  deviate_pub_be_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "/wall_plane_filter/cloud_deviate", cloud_be);
  lidar_wall_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    "/wall_plane_filter/lidar_wall_markers", rclcpp::QoS(1).reliable());
  depth_wall_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    "/wall_plane_filter/depth_wall_markers", rclcpp::QoS(1).reliable());
  lidar_wall_plane_pub_ = create_publisher<wall_plane_filter::msg::WallPlaneArray>(
    "/wall_plane_filter/lidar_wall_planes", rclcpp::QoS(1).reliable());
  depth_wall_plane_pub_ = create_publisher<wall_plane_filter::msg::WallPlaneArray>(
    "/wall_plane_filter/depth_wall_planes", rclcpp::QoS(1).reliable());
  elevator_door_pub_ = create_publisher<wall_plane_filter::msg::ElevatorDoorStatus>(
    "/wall_plane_filter/elevator_door_status", rclcpp::QoS(1).reliable().transient_local());

  if (params_.depth_hough.debug) {
    depth_grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/wall_plane_filter/depth_hough/binary", rclcpp::QoS(1).reliable());
    depth_debug_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/wall_plane_filter/depth_hough/debug_markers", rclcpp::QoS(1).reliable());
  }
  if (params_.lidar_hough.debug) {
    lidar_grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/wall_plane_filter/lidar_hough/binary", rclcpp::QoS(1).reliable());
    lidar_debug_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/wall_plane_filter/lidar_hough/debug_markers", rclcpp::QoS(1).reliable());
  }

  const auto lidar_cb = std::bind(&WallPlaneFilterNode::cloudCallback, this, std::placeholders::_1);
  cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    params_.cloud_topic, rclcpp::SensorDataQoS().keep_last(2).reliable(), lidar_cb);
  cloud_sub_compat_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    params_.cloud_topic, rclcpp::SensorDataQoS().keep_last(2).best_effort(), lidar_cb);

  for (const auto & topic : params_.depth_topics) {
    if (topic.empty() || !params_.depth_clean_enable) {
      continue;
    }
    const auto slash = topic.find('/', 1);
    std::string cam = "camera";
    if (slash != std::string::npos && slash > 1) {
      cam = topic.substr(1, slash - 1);
    }
    bool find_walls = false;
    for (const auto & name : params_.depth_wall_cameras) {
      if (name == cam) {
        find_walls = true;
        break;
      }
    }
    const auto cloud_rel = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
    const auto cloud_be = rclcpp::SensorDataQoS().keep_last(5);
    const std::string filtered_topic = "/wall_plane_filter/" + cam + "/depth_cloud_filtered";
    const std::string deviate_topic = "/wall_plane_filter/" + cam + "/depth_cloud_deviate";
    depth_filtered_pubs_.push_back(create_publisher<sensor_msgs::msg::PointCloud2>(
      filtered_topic, cloud_rel));
    depth_filtered_pubs_be_.push_back(create_publisher<sensor_msgs::msg::PointCloud2>(
      filtered_topic, cloud_be));
    depth_deviate_pubs_.push_back(create_publisher<sensor_msgs::msg::PointCloud2>(
      deviate_topic, cloud_rel));
    depth_deviate_pubs_be_.push_back(create_publisher<sensor_msgs::msg::PointCloud2>(
      deviate_topic, cloud_be));
    depth_cam_plane_pubs_.push_back(create_publisher<wall_plane_filter::msg::WallPlaneArray>(
      "/wall_plane_filter/" + cam + "/wall_planes", rclcpp::QoS(1).reliable()));
    depth_cam_names_.push_back(cam);
    depth_find_walls_.push_back(find_walls ? 1 : 0);
    depth_slots_.emplace_back();
    depth_cam_tracks_.emplace_back();
    depth_grid_origins_.emplace_back(Eigen::Vector2d::Zero());
    depth_grid_valid_.push_back(0);
    depth_tf_total_.push_back(0);
    depth_tf_fail_.push_back(0);
    const size_t idx = depth_slots_.size() - 1;
    const auto depth_cb = [this, idx](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      this->depthCallback(msg, idx);
    };
    depth_subs_.push_back(create_subscription<sensor_msgs::msg::PointCloud2>(
      topic, rclcpp::SensorDataQoS().keep_last(2).reliable(), depth_cb));
    depth_subs_.push_back(create_subscription<sensor_msgs::msg::PointCloud2>(
      topic, rclcpp::SensorDataQoS().keep_last(2).best_effort(), depth_cb));
  }

  RCLCPP_INFO(
    get_logger(),
    "lidar z-cut [%.2f, %.2f] range<%.2f wall_edge=%d cloud=%s depth_walls=per_camera",
    params_.lidar_z_cut_down, params_.lidar_z_cut_up, params_.lidar_z_cut_range_max,
    params_.wall_edge_enable ? 1 : 0, params_.cloud_topic.c_str());
}

void WallPlaneFilterNode::loadParams()
{
  const Params d;
  auto take = [this](const std::string & name, auto & dst, const auto & fallback) {
    if (!this->has_parameter(name)) {
      this->declare_parameter(name, fallback);
    }
    this->get_parameter(name, dst);
  };
  take("cloud_topic", params_.cloud_topic, d.cloud_topic);
  take("target_frame", params_.target_frame, d.target_frame);
  take("lidar_frame", params_.lidar_frame, d.lidar_frame);
  take("z_min", params_.z_min, d.z_min);
  take("z_max", params_.z_max, d.z_max);
  take("range_min", params_.range_min, d.range_min);
  take("range_max", params_.range_max, d.range_max);
  take("voxel_leaf", params_.voxel_leaf, d.voxel_leaf);
  take("process_hz", params_.process_hz, d.process_hz);
  take("tf_timeout", params_.tf_timeout, d.tf_timeout);
  take("lidar_sub_reliable", params_.lidar_sub_reliable, d.lidar_sub_reliable);
  take("lidar_z_cut_enable", params_.lidar_z_cut_enable, d.lidar_z_cut_enable);
  take("lidar_z_cut_down", params_.lidar_z_cut_down, d.lidar_z_cut_down);
  take("lidar_z_cut_up", params_.lidar_z_cut_up, d.lidar_z_cut_up);
  take("lidar_z_cut_range_max", params_.lidar_z_cut_range_max, d.lidar_z_cut_range_max);
  take("lidar_wall_enable", params_.lidar_wall_enable, d.lidar_wall_enable);
  take("lidar_wall_range_max", params_.lidar_wall_range_max, d.lidar_wall_range_max);
  take("wall_edge_enable", params_.wall_edge_enable, d.wall_edge_enable);
  take("edge_min", params_.edge_min, d.edge_min);
  take("edge_max", params_.edge_max, d.edge_max);
  take("ring_bow_half_deg", params_.ring_bow_half_deg, d.ring_bow_half_deg);
  take("ap_el_res_deg", params_.ap_el_res_deg, d.ap_el_res_deg);
  take("ap_keep_depth", params_.ap_keep_depth, d.ap_keep_depth);
  take("depth_clean_enable", params_.depth_clean_enable, d.depth_clean_enable);
  take("depth_topics", params_.depth_topics, d.depth_topics);
  take("depth_sub_reliable", params_.depth_sub_reliable, d.depth_sub_reliable);
  take("depth_z_min", params_.depth_z_min, d.depth_z_min);
  take("depth_z_max", params_.depth_z_max, d.depth_z_max);
  take("depth_range_min", params_.depth_range_min, d.depth_range_min);
  take("depth_range_max", params_.depth_range_max, d.depth_range_max);
  take("depth_voxel_leaf", params_.depth_voxel_leaf, d.depth_voxel_leaf);
  take("depth_publish_voxel", params_.depth_publish_voxel, d.depth_publish_voxel);
  take("depth_wall_band_m", params_.depth_wall_band_m, d.depth_wall_band_m);
  take("depth_wall_band_outer_m", params_.depth_wall_band_outer_m, d.depth_wall_band_outer_m);
  take("depth_wall_project", params_.depth_wall_project, d.depth_wall_project);
  take("depth_local_flatten_enable", params_.depth_local_flatten_enable, d.depth_local_flatten_enable);
  take("depth_local_flatten_radius", params_.depth_local_flatten_radius, d.depth_local_flatten_radius);
  take(
    "depth_local_flatten_min_neighbors", params_.depth_local_flatten_min_neighbors,
    d.depth_local_flatten_min_neighbors);
  take(
    "depth_local_flatten_max_curvature", params_.depth_local_flatten_max_curvature,
    d.depth_local_flatten_max_curvature);
  take(
    "depth_local_flatten_max_points", params_.depth_local_flatten_max_points,
    d.depth_local_flatten_max_points);
  take("depth_fly_enable", params_.depth_fly_enable, d.depth_fly_enable);
  take("depth_fly_abs_m", params_.depth_fly_abs_m, d.depth_fly_abs_m);
  take("depth_fly_rel", params_.depth_fly_rel, d.depth_fly_rel);
  take("depth_fly_min_neighbors", params_.depth_fly_min_neighbors, d.depth_fly_min_neighbors);
  params_.depth_fly_min_neighbors = std::clamp(params_.depth_fly_min_neighbors, 1, 8);
  take("depth_median_enable", params_.depth_median_enable, d.depth_median_enable);
  take("depth_median_half", params_.depth_median_half, d.depth_median_half);
  take("depth_median_min", params_.depth_median_min, d.depth_median_min);
  take("depth_median_edge_mad", params_.depth_median_edge_mad, d.depth_median_edge_mad);
  take("depth_decimate_stride", params_.depth_decimate_stride, d.depth_decimate_stride);
  take("wall_project_local_check", params_.wall_project_local_check, d.wall_project_local_check);
  take("wall_project_radius", params_.wall_project_radius, d.wall_project_radius);
  take("wall_project_normal_deg", params_.wall_project_normal_deg, d.wall_project_normal_deg);
  take(
    "wall_project_max_curvature", params_.wall_project_max_curvature,
    d.wall_project_max_curvature);
  take(
    "wall_project_min_neighbors", params_.wall_project_min_neighbors,
    d.wall_project_min_neighbors);
  take(
    "wall_fuse_min_support_points", params_.wall_fuse_min_support_points,
    d.wall_fuse_min_support_points);
  take(
    "depth_cull_use_global_walls", params_.depth_cull_use_global_walls,
    d.depth_cull_use_global_walls);
  take("depth_wall_range_max", params_.depth_wall_range_max, d.depth_wall_range_max);
  take("depth_wall_fuse_hz", params_.depth_wall_fuse_hz, d.depth_wall_fuse_hz);
  take("depth_wall_cameras", params_.depth_wall_cameras, d.depth_wall_cameras);
  take("depth_merge_max_age", params_.depth_merge_max_age, d.depth_merge_max_age);
  take("depth_wall_noise_k", params_.depth_wall_noise_k, d.depth_wall_noise_k);
  take("depth_wall_noise_max", params_.depth_wall_noise_max, d.depth_wall_noise_max);
  take("depth_wall_ground_margin", params_.depth_wall_ground_margin,
    d.depth_wall_ground_margin);
  take("depth_wall_tf_report_sec", params_.depth_wall_tf_report_sec,
    d.depth_wall_tf_report_sec);
  take("depth_wall_map_frame", params_.depth_wall_map_frame, d.depth_wall_map_frame);
  take("depth_wall_output_frame", params_.depth_wall_output_frame, d.depth_wall_output_frame);
  take("depth_ground_ref_z", params_.depth_ground_ref_z, d.depth_ground_ref_z);
  take("depth_radius_enable", params_.depth_radius_enable, d.depth_radius_enable);
  take("depth_radius_search", params_.depth_radius_search, d.depth_radius_search);
  take("depth_radius_min_neighbors", params_.depth_radius_min_neighbors, d.depth_radius_min_neighbors);
  auto hough = [&](const std::string & prefix, HoughConfig & config) {
    const HoughConfig defaults;
    take(prefix + ".resolution", config.resolution, defaults.resolution);
    take(prefix + ".half_extent", config.half_extent, defaults.half_extent);
    take(prefix + ".follow_sensor", config.follow_sensor, defaults.follow_sensor);
    take(prefix + ".origin_x", config.origin_x, defaults.origin_x);
    take(prefix + ".origin_y", config.origin_y, defaults.origin_y);
    take(prefix + ".max_grid_side", config.max_grid_side, defaults.max_grid_side);
    take(prefix + ".min_cell_points", config.min_cell_points, defaults.min_cell_points);
    take(prefix + ".min_cell_height", config.min_cell_height, defaults.min_cell_height);
    take(prefix + ".z_min", config.z_min, defaults.z_min);
    take(prefix + ".z_max", config.z_max, defaults.z_max);
    take(prefix + ".theta_deg", config.theta_deg, defaults.theta_deg);
    take(prefix + ".votes", config.votes, defaults.votes);
    take(prefix + ".min_line_length", config.min_line_length, defaults.min_line_length);
    take(prefix + ".hough_max_gap", config.hough_max_gap, defaults.hough_max_gap);
    take(prefix + ".merge_angle_deg", config.merge_angle_deg, defaults.merge_angle_deg);
    take(prefix + ".merge_distance", config.merge_distance, defaults.merge_distance);
    take(prefix + ".merge_gap", config.merge_gap, defaults.merge_gap);
    take(prefix + ".parallel_angle_deg", config.parallel_angle_deg, defaults.parallel_angle_deg);
    take(prefix + ".intersection_extension", config.intersection_extension, defaults.intersection_extension);
    take(prefix + ".support_band", config.support_band, defaults.support_band);
    take(prefix + ".min_support_ratio", config.min_support_ratio, defaults.min_support_ratio);
    take(prefix + ".max_unsupported_gap", config.max_unsupported_gap, defaults.max_unsupported_gap);
    take(prefix + ".min_continuous_support", config.min_continuous_support, defaults.min_continuous_support);
    take(prefix + ".min_support_points", config.min_support_points, defaults.min_support_points);
    take(prefix + ".max_lines", config.max_lines, defaults.max_lines);
    take(prefix + ".refine", config.refine, defaults.refine);
    take(prefix + ".debug", config.debug, defaults.debug);
    validateHoughConfig(config);
  };
  hough("depth_hough", params_.depth_hough);
  hough("lidar_hough", params_.lidar_hough);
  {
    WallTrackConfig & t = params_.wall_track;
    const WallTrackConfig td;
    take("wall_track.normal_gate", t.normal_gate, td.normal_gate);
    take("wall_track.merge_gap", t.merge_gap, td.merge_gap);
    take("wall_track.obs_weight", t.obs_weight, td.obs_weight);
    take("wall_track.confirm_frames", t.confirm_frames, td.confirm_frames);
    take("wall_track.shorten_frames", t.shorten_frames, td.shorten_frames);
    take("wall_track.lose_sec", t.lose_sec, td.lose_sec);
    if (!(t.lose_sec > 0.0) || t.confirm_frames < 1 || t.shorten_frames < 1 ||
      !(t.normal_gate > 0.0) || !(t.merge_gap >= 0.0) ||
      t.obs_weight < 0.0 || t.obs_weight > 1.0)
    {
      throw std::invalid_argument("Invalid wall_track temporal parameters");
    }
  }
  take("elevator_door_enable", params_.elevator_door_enable, d.elevator_door_enable);
  take("elevator_door_x_min", params_.elevator_door_x_min, d.elevator_door_x_min);
  take("elevator_door_x_max", params_.elevator_door_x_max, d.elevator_door_x_max);
  take("elevator_door_half_width", params_.elevator_door_half_width, d.elevator_door_half_width);
  take("elevator_door_z_min", params_.elevator_door_z_min, d.elevator_door_z_min);
  take("elevator_door_z_max", params_.elevator_door_z_max, d.elevator_door_z_max);
  take("elevator_door_open_margin", params_.elevator_door_open_margin, d.elevator_door_open_margin);
  take("elevator_door_open_ratio", params_.elevator_door_open_ratio, d.elevator_door_open_ratio);
  take("elevator_door_bins", params_.elevator_door_bins, d.elevator_door_bins);
  take("elevator_door_min_points_bin", params_.elevator_door_min_points_bin,
    d.elevator_door_min_points_bin);
  take("elevator_door_min_valid_bins", params_.elevator_door_min_valid_bins,
    d.elevator_door_min_valid_bins);
  take("elevator_door_vote_frames", params_.elevator_door_vote_frames, d.elevator_door_vote_frames);
  take("wall_line_width", params_.wall_line_width, d.wall_line_width);
  take("wall_display_z", params_.wall_display_z, d.wall_display_z);
  take("wall_marker_lifetime", params_.wall_marker_lifetime, d.wall_marker_lifetime);
  if (!std::isfinite(params_.depth_wall_fuse_hz) || params_.depth_wall_fuse_hz <= 0 || params_.depth_wall_fuse_hz > 30 ||
    !std::isfinite(params_.depth_merge_max_age) || params_.depth_merge_max_age <= 0 ||
    !std::isfinite(params_.wall_line_width) || params_.wall_line_width <= 0 ||
    !std::isfinite(params_.wall_display_z) || !std::isfinite(params_.wall_marker_lifetime) || params_.wall_marker_lifetime <= 0)
  {
    throw std::invalid_argument("Invalid wall publication rate, age, width, height or lifetime");
  }

}

bool WallPlaneFilterNode::cloudToTarget(
  const sensor_msgs::msg::PointCloud2 & msg, CloudT & out, rclcpp::Time & stamp)
{
  stamp = rclcpp::Time(msg.header.stamp);
  CloudT raw;
  pcl::fromROSMsg(msg, raw);
  return transformCloud(raw, params_.target_frame, msg.header.frame_id, stamp, out);
}

bool WallPlaneFilterNode::lookupIso(
  const std::string & to_frame, const std::string & from_frame, const rclcpp::Time & stamp,
  Eigen::Isometry3d & out)
{
  if (to_frame.empty() || from_frame.empty()) {
    return false;
  }
  if (to_frame == from_frame) {
    out = Eigen::Isometry3d::Identity();
    return true;
  }
  try {
    out = tf2::transformToEigen(tf_buffer_->lookupTransform(
      to_frame, from_frame, stamp, tf2::durationFromSec(params_.tf_timeout)));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "TF %s<-%s failed: %s",
      to_frame.c_str(), from_frame.c_str(), ex.what());
    return false;
  }
  return true;
}

bool WallPlaneFilterNode::transformCloud(
  const CloudT & in, const std::string & to_frame, const std::string & from_frame,
  const rclcpp::Time & stamp, CloudT & out)
{
  out.clear();
  if (to_frame == from_frame && !to_frame.empty()) {
    out = in;
    return true;
  }
  Eigen::Isometry3d T;
  if (!lookupIso(to_frame, from_frame, stamp, T)) {
    return false;
  }
  out.reserve(in.size());
  for (const auto & p : in.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      continue;
    }
    const Eigen::Vector3d q = T * Eigen::Vector3d(p.x, p.y, p.z);
    out.push_back(pcl::PointXYZ(
      static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z())));
  }
  out.width = static_cast<uint32_t>(out.size());
  out.height = 1;
  out.is_dense = false;
  return true;
}

std::string WallPlaneFilterNode::wallWorkFrame() const
{
  if (!params_.depth_wall_map_frame.empty()) {
    return params_.depth_wall_map_frame;
  }
  return params_.target_frame;
}

std::string WallPlaneFilterNode::outputFrame() const
{
  std::string out = params_.depth_wall_output_frame.empty() ?
    wallWorkFrame() : params_.depth_wall_output_frame;
  if (out == "base_link") {
    out = params_.target_frame;
  }
  return out;
}

void WallPlaneFilterNode::clearWallCachesOnTimeJump(const rclcpp::Time & stamp)
{
  // Large PointCloud2 streams may be delivered seconds out of stamp order.
  // Detect bag seek/loop from the node's simulation clock, never from a late
  // sensor message. Message stamps remain authoritative for every TF lookup.
  rclcpp::Time reference = get_clock()->now();
  if (reference.nanoseconds() == 0) {
    reference = stamp;
  }
  const double tol = std::max(1.0, 2.0 * params_.depth_merge_max_age);
  if (wall_path_stamp_.nanoseconds() == 0) {
    wall_path_stamp_ = reference;
    return;
  }
  const double dt = (reference - wall_path_stamp_).seconds();
  if (dt >= 0.0) {
    wall_path_stamp_ = reference;
    return;
  }
  if (-dt <= tol) {
    return;
  }
  RCLCPP_WARN(get_logger(),
    "wall path stamp jumped backward %.3fs; clearing camera slots and validated Hough lines",
    -dt);
  std::lock_guard<std::mutex> lock(depth_wall_mutex_);
  for (auto & slot : depth_slots_) {
    slot = DepthSlot();
  }
  visualization_msgs::msg::MarkerArray reset;
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = outputFrame();
  marker.header.stamp = stamp;
  marker.action = visualization_msgs::msg::Marker::DELETEALL;
  reset.markers.push_back(marker);
  depth_wall_marker_pub_->publish(reset);
  lidar_wall_marker_pub_->publish(reset);
  if (depth_debug_pub_) {depth_debug_pub_->publish(reset);}
  if (lidar_debug_pub_) {lidar_debug_pub_->publish(reset);}
  last_depth_wall_markers_ = visualization_msgs::msg::MarkerArray();
  last_lidar_wall_markers_ = visualization_msgs::msg::MarkerArray();
  wall_plane_filter::msg::WallPlaneArray empty_planes;
  empty_planes.header = marker.header;
  depth_wall_plane_pub_->publish(empty_planes);
  lidar_wall_plane_pub_->publish(empty_planes);
  last_depth_wall_planes_ = empty_planes;
  last_lidar_wall_planes_ = empty_planes;
  for (auto & pub : depth_cam_plane_pubs_) {
    pub->publish(empty_planes);
  }
  validated_depth_lines_.clear();
  validated_depth_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  for (auto & tracker : depth_cam_tracks_) {
    tracker.clear();
  }
  lidar_tracks_.clear();
  for (size_t i = 0; i < depth_grid_valid_.size(); ++i) {
    depth_grid_valid_[i] = 0;
    depth_grid_origins_[i] = Eigen::Vector2d::Zero();
  }
  wall_path_stamp_ = reference;
}

void WallPlaneFilterNode::filterRoi(CloudT & cloud) const
{
  CloudT kept;
  kept.reserve(cloud.size());
  for (const auto & p : cloud.points) {
    if (p.z < params_.z_min || p.z > params_.z_max) {
      continue;
    }
    const double r = std::hypot(p.x, p.y);
    if (r < params_.range_min || r > params_.range_max) {
      continue;
    }
    kept.push_back(p);
  }
  cloud.swap(kept);
  cloud.width = static_cast<uint32_t>(cloud.size());
  cloud.height = 1;
}

void WallPlaneFilterNode::voxelDownsample(CloudT & cloud) const
{
  if (cloud.empty() || params_.voxel_leaf <= 1e-4) {
    return;
  }
  CloudT out;
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cloud.makeShared());
  const float leaf = static_cast<float>(params_.voxel_leaf);
  vg.setLeafSize(leaf, leaf, leaf);
  vg.filter(out);
  cloud.swap(out);
}

sensor_msgs::msg::PointCloud2 WallPlaneFilterNode::toMsg(
  const CloudT & cloud, const rclcpp::Time & stamp) const
{
  return toMsg(cloud, stamp, params_.target_frame);
}

sensor_msgs::msg::PointCloud2 WallPlaneFilterNode::toMsg(
  const CloudT & cloud, const rclcpp::Time & stamp, const std::string & frame) const
{
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  msg.header.frame_id = frame.empty() ? params_.target_frame : frame;
  msg.header.stamp = stamp;
  return msg;
}

void WallPlaneFilterNode::publishCloud(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & reliable,
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & best_effort,
  const sensor_msgs::msg::PointCloud2 & msg, const char * label)
{
  size_t subs = 0;
  if (reliable) {
    reliable->publish(msg);
    subs += reliable->get_subscription_count();
  }
  if (best_effort) {
    best_effort->publish(msg);
    subs += best_effort->get_subscription_count();
  }
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 2000,
    "publish %s frame=%s points≈%u bytes=%zu matched_subs=%zu",
    label, msg.header.frame_id.c_str(),
    msg.width * msg.height, msg.data.size(), subs);
}

bool WallPlaneFilterNode::acceptStampOnce(int key, int64_t stamp_ns)
{
  std::lock_guard<std::mutex> lock(dedup_mutex_);
  for (const auto & item : recent_stamps_) {
    if (item.first == key && item.second == stamp_ns) {
      return false;
    }
  }
  recent_stamps_.emplace_back(key, stamp_ns);
  while (recent_stamps_.size() > kDedupHistory) {
    recent_stamps_.pop_front();
  }
  return true;
}

void WallPlaneFilterNode::cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  if (!msg) {
    return;
  }
  const rclcpp::Time stamp(msg->header.stamp);
  clearWallCachesOnTimeJump(stamp);
  if (!acceptStampOnce(-1, stamp.nanoseconds())) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(stamp_mutex_);
    if (params_.process_hz > 1e-3 && last_stamp_.nanoseconds() > 0) {
      const double dt = (stamp - last_stamp_).seconds();
      if (dt >= 0.0 && dt < 1.0 / params_.process_hz) {
        return;
      }
    }
    last_stamp_ = stamp;
  }

  CloudT cloud;
  rclcpp::Time cloud_stamp;
  if (!cloudToTarget(*msg, cloud, cloud_stamp)) {
    return;
  }
  const size_t input_n = cloud.size();
  filterRoi(cloud);
  const size_t roi_n = cloud.size();

  const std::string src = params_.lidar_frame.empty() ? msg->header.frame_id : params_.lidar_frame;
  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  Eigen::Vector3d z_axis = Eigen::Vector3d::UnitZ();
  try {
    const auto tf = tf_buffer_->lookupTransform(
      params_.target_frame, src, cloud_stamp, tf2::durationFromSec(params_.tf_timeout));
    const Eigen::Isometry3d T = tf2::transformToEigen(tf);
    origin = T.translation();
    z_axis = T.rotation() * Eigen::Vector3d::UnitZ();
  } catch (const tf2::TransformException &) {
  }

  CloudT filtered, deviate;
  runLidarTailPipeline(cloud, origin, z_axis, params_, filtered, deviate);
  const size_t denoised_n = filtered.size();
  voxelDownsample(filtered);
  const size_t deviate_n = deviate.size();

  // Cull/project using previously confirmed lidar walls (map → target_frame).
  {
    std::vector<WallLine> map_walls;
    {
      std::lock_guard<std::mutex> lock(depth_wall_mutex_);
      map_walls = lidar_tracks_.confirmed();
    }
    if (!map_walls.empty()) {
      const std::string map_frame = wallWorkFrame();
      Eigen::Isometry3d to_target = Eigen::Isometry3d::Identity();
      if (lookupIso(params_.target_frame, map_frame, cloud_stamp, to_target)) {
        const std::vector<WallLine> walls_base = transformLines(map_walls, to_target);
        const Eigen::Vector2d sensor_xy = origin.head<2>();
        cullWallBand(
          filtered, deviate, walls_base,
          params_.depth_wall_band_m, params_.depth_wall_band_outer_m,
          params_.depth_wall_project, Eigen::Vector2d::Zero(),
          params_.depth_wall_noise_k, params_.depth_wall_noise_max, nullptr, &sensor_xy,
          &params_);
      }
    }
  }

  // Display clouds go out immediately; wall tracking does not gate them.
  publishLidarCloud(filtered, deviate, cloud_stamp);

  if (params_.lidar_wall_enable) {
    CloudT near;
    for (const auto & p : filtered) {
      if (std::hypot(p.x, p.y) <= params_.lidar_wall_range_max &&
        p.z >= params_.lidar_hough.z_min && p.z <= params_.lidar_hough.z_max)
      {
        near.push_back(p);
      }
    }
    const auto result = extractHoughWalls(near, Eigen::Vector2d::Zero(), params_.lidar_hough);
    if (params_.lidar_hough.debug) {
      publishHoughDebug(result, cloud_stamp, params_.target_frame, false);
    }
    // Fit stays in the lidar frame. Tracking runs only after the stamp's TF
    // into map, so the smoother never blends base_link motion.
    const std::string map_frame = wallWorkFrame();
    Eigen::Isometry3d to_map = Eigen::Isometry3d::Identity();
    std::vector<WallLine> map_obs;
    std::vector<Eigen::Vector2d> map_corners;
    if (lookupIso(map_frame, params_.target_frame, cloud_stamp, to_map)) {
      map_obs = transformLines(result.lines, to_map);
      map_corners = transformPoints(result.corners, to_map);
    }
    std::vector<WallLine> tracked;
    {
      std::lock_guard<std::mutex> lock(depth_wall_mutex_);
      tracked = lidar_tracks_.update(
        map_obs, cloud_stamp.seconds(), trackConfigFrom(params_.lidar_hough, params_.wall_track));
    }
    publishTrackedMarkers(tracked, map_corners, cloud_stamp, map_frame, false);
  }

  updateElevatorDoor(filtered, cloud_stamp);

  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 2000,
    "lidar preprocess input=%zu roi=%zu denoised=%zu deviate=%zu voxel=%zu",
    input_n, roi_n, denoised_n, deviate_n, filtered.size());
}

void WallPlaneFilterNode::depthCallback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg, size_t idx)
{
  if (!msg || idx >= depth_slots_.size()) {
    return;
  }
  const rclcpp::Time stamp(msg->header.stamp);
  clearWallCachesOnTimeJump(stamp);
  if (!acceptStampOnce(static_cast<int>(idx), stamp.nanoseconds())) {
    return;
  }
  const bool find_walls = depth_find_walls_[idx] != 0;
  const std::string world = wallWorkFrame();
  // A zero stamp cannot be looked up. Use the latest TF only in that case.
  const rclcpp::Time lookup_stamp = stamp.nanoseconds() == 0 ?
    rclcpp::Time(0, 0, get_clock()->get_clock_type()) : stamp;

  {
    std::lock_guard<std::mutex> lock(depth_tf_mutex_);
    ++depth_tf_total_[idx];
  }
  Eigen::Isometry3d T_bc = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_wb = Eigen::Isometry3d::Identity();
  const bool base_ok = lookupIso(params_.target_frame, msg->header.frame_id, lookup_stamp, T_bc);
  const bool map_ok = base_ok && lookupIso(world, params_.target_frame, lookup_stamp, T_wb);
  if (!map_ok) {
    std::lock_guard<std::mutex> lock(depth_tf_mutex_);
    ++depth_tf_fail_[idx];
  }
  reportDepthTf();
  if (!base_ok) {
    return;
  }

  CloudT raw;
  pcl::fromROSMsg(*msg, raw);
  CloudT image_kept;
  CloudT fly_cam;
  denoiseDepthImage(raw, params_, image_kept, fly_cam);

  CloudT roi;
  CloudT deviate;
  keepInRoi(image_kept, T_bc, params_, roi);
  keepInRoi(fly_cam, T_bc, params_, deviate);

  const double publish_leaf = params_.depth_publish_voxel > 1e-4 ?
    params_.depth_publish_voxel : params_.depth_voxel_leaf;
  CloudT voxel;
  voxelLeaf(roi, publish_leaf, voxel);
  const size_t voxel_n = voxel.size();

  CloudT filtered;
  size_t radius_drop = 0;
  if (params_.depth_radius_enable && voxel_n > 0) {
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror(true);
    ror.setInputCloud(voxel.makeShared());
    ror.setRadiusSearch(params_.depth_radius_search);
    ror.setMinNeighborsInRadius(std::max(1, params_.depth_radius_min_neighbors));
    ror.filter(filtered);
    const auto removed = ror.getRemovedIndices();
    if (removed) {
      radius_drop = removed->size();
      deviate.reserve(deviate.size() + radius_drop);
      for (const auto i : *removed) {
        deviate.push_back(voxel.points[static_cast<size_t>(i)]);
      }
      finishCloud(deviate);
    }
  } else {
    filtered = std::move(voxel);
  }
  finishCloud(filtered);
  // Door ROI is defined in target_frame (base_link); filtered is already there.
  if (depth_cam_names_[idx].find("front") != std::string::npos ||
    depth_cam_names_[idx] == "camera1")
  {
    updateElevatorDoor(filtered, stamp);
  }
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 2000,
    "depth preprocess camera=%s input=%zu fly=%zu image=%zu roi=%zu voxel=%zu "
    "radius_drop=%zu out=%zu",
    depth_cam_names_[idx].c_str(), raw.size(), fly_cam.size(), image_kept.size(),
    roi.size(), voxel_n, radius_drop, filtered.size());
  if (!map_ok) {
    // base_link is available, so RViz can still draw the cloud once map TF arrives.
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "depth camera=%s %s<-%s failed; publishing filtered cloud in %s",
      depth_cam_names_[idx].c_str(), world.c_str(), params_.target_frame.c_str(),
      params_.target_frame.c_str());
    publishCloud(
      depth_filtered_pubs_[idx], depth_filtered_pubs_be_[idx],
      toMsg(filtered, stamp, params_.target_frame), "depth_filtered");
    publishCloud(
      depth_deviate_pubs_[idx], depth_deviate_pubs_be_[idx],
      toMsg(deviate, stamp, params_.target_frame), "depth_deviate");
    return;
  }
  DepthSlot slot;
  transformInto(filtered, T_wb, slot.filtered_world);
  transformInto(deviate, T_wb, slot.deviate_world);
  slot.robot_xy = T_wb.translation().head<2>();
  slot.sensor_xy = (T_wb * T_bc).translation().head<2>();
  slot.stamp = stamp;

  // Publish first with cached walls (latency), then update Hough tracks.
  publishDepthCloud(idx, slot);

  if (find_walls) {
    // Wall-finding subset: above the floor, near, and within trusted depth noise.
    const double floor_cut = params_.depth_ground_ref_z + params_.depth_wall_ground_margin;
    const Eigen::Vector3d cam_b = T_bc.translation();
    const double noise_k = std::max(0.0, params_.depth_wall_noise_k);
    CloudT evidence;
    if (params_.depth_voxel_leaf > publish_leaf + 1e-4) {
      voxelLeaf(filtered, params_.depth_voxel_leaf, evidence);
    } else {
      evidence = filtered;
    }
    CloudT wall_base;
    wall_base.reserve(evidence.size());
    for (const auto & p : evidence.points) {
      if (p.z < std::max(floor_cut, params_.depth_hough.z_min) ||
        p.z > params_.depth_hough.z_max || std::hypot(p.x, p.y) > params_.depth_wall_range_max)
      {
        continue;
      }
      const double d = (Eigen::Vector3d(p.x, p.y, p.z) - cam_b).norm();
      if (noise_k * d * d > params_.depth_wall_noise_max) {
        continue;
      }
      wall_base.push_back(p);
    }
    finishCloud(wall_base);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "depth wall subset camera=%s voxel=%zu eligible=%zu floor_cut=%.2f range=%.2f",
      depth_cam_names_[idx].c_str(), evidence.size(), wall_base.size(), floor_cut,
      params_.depth_wall_range_max);
    CloudT wall_world;
    transformInto(wall_base, T_wb, wall_world);
    processDepthWalls(idx, wall_world, slot.robot_xy, stamp);
  }

  std::lock_guard<std::mutex> lock(depth_wall_mutex_);
  depth_slots_[idx] = std::move(slot);
}

void WallPlaneFilterNode::publishLidarCloud(
  const CloudT & filtered, const CloudT & deviate, const rclcpp::Time & stamp)
{
  CloudT filtered_out, deviate_out;
  const std::string out = outputFrame();
  if (transformCloud(filtered, out, params_.target_frame, stamp, filtered_out) &&
    transformCloud(deviate, out, params_.target_frame, stamp, deviate_out))
  {
    publishCloud(filtered_pub_, filtered_pub_be_, toMsg(filtered_out, stamp, out), "lidar_filtered");
    publishCloud(deviate_pub_, deviate_pub_be_, toMsg(deviate_out, stamp, out), "lidar_deviate");
    return;
  }
  // Still publish in target_frame so RViz can draw once Fixed Frame TF is ready.
  publishCloud(
    filtered_pub_, filtered_pub_be_, toMsg(filtered, stamp, params_.target_frame), "lidar_filtered");
  publishCloud(
    deviate_pub_, deviate_pub_be_, toMsg(deviate, stamp, params_.target_frame), "lidar_deviate");
}

void WallPlaneFilterNode::publishDepthCloud(size_t idx, const DepthSlot & slot)
{
  std::vector<WallLine> walls;
  {
    std::lock_guard<std::mutex> lock(depth_wall_mutex_);
    const bool stamp_ok = validated_depth_stamp_.nanoseconds() != 0 &&
      std::abs((slot.stamp - validated_depth_stamp_).seconds()) <=
      std::max(params_.depth_merge_max_age, 0.5);
    if (idx < depth_cam_tracks_.size()) {
      walls = depth_cam_tracks_[idx].confirmed();
    }
    if (params_.depth_cull_use_global_walls) {
      if (stamp_ok && !validated_depth_lines_.empty()) {
        walls = validated_depth_lines_;
      }
    } else if (walls.empty() && stamp_ok) {
      // Prefer local tracks; fall back to strong global lines only.
      for (const auto & line : validated_depth_lines_) {
        if (static_cast<int>(line.support_points) >= params_.wall_fuse_min_support_points) {
          walls.push_back(line);
        }
      }
    }
  }
  CloudT filtered = slot.filtered_world;
  CloudT deviate = slot.deviate_world;
  size_t band_drop = 0;
  size_t band_proj = 0;
  size_t flat_n = 0;
  if (!walls.empty()) {
    band_drop = cullWallBand(
      filtered, deviate, walls,
      params_.depth_wall_band_m, params_.depth_wall_band_outer_m,
      params_.depth_wall_project, slot.robot_xy,
      params_.depth_wall_noise_k, params_.depth_wall_noise_max, &band_proj,
      &slot.sensor_xy, &params_);
  } else if (params_.depth_local_flatten_enable) {
    // No wall yet: PCA flatten only. Skip when walls exist — wall project is enough
    // and full-cloud PCA (~5k radius searches) was starving depth Hough callbacks.
    flat_n = flattenLocalPlanes(
      filtered, params_.depth_local_flatten_radius,
      params_.depth_local_flatten_min_neighbors, params_.depth_local_flatten_max_curvature,
      params_.depth_local_flatten_max_points);
  }
  if (band_drop > 0 || band_proj > 0 || flat_n > 0) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "depth flatten camera=%s walls=%zu drop=%zu proj=%zu local_flat=%zu project=%s band=%.3f outer=%.3f",
      depth_cam_names_[idx].c_str(), walls.size(), band_drop, band_proj, flat_n,
      params_.depth_wall_project ? "on" : "off",
      params_.depth_wall_band_m, params_.depth_wall_band_outer_m);
  }
  const std::string world = wallWorkFrame();
  const std::string out_frame = outputFrame();
  CloudT filtered_out;
  CloudT deviate_out;
  if (transformCloud(filtered, out_frame, world, slot.stamp, filtered_out) &&
    transformCloud(deviate, out_frame, world, slot.stamp, deviate_out))
  {
    publishCloud(
      depth_filtered_pubs_[idx], depth_filtered_pubs_be_[idx],
      toMsg(filtered_out, slot.stamp, out_frame), "depth_filtered");
    publishCloud(
      depth_deviate_pubs_[idx], depth_deviate_pubs_be_[idx],
      toMsg(deviate_out, slot.stamp, out_frame), "depth_deviate");
    return;
  }
  RCLCPP_WARN_THROTTLE(
    get_logger(), *get_clock(), 2000,
    "depth cloud camera=%s TF %s<-%s stamp=%.6f failed; publishing in %s",
    depth_cam_names_[idx].c_str(), out_frame.c_str(), world.c_str(), slot.stamp.seconds(),
    world.c_str());
  publishCloud(
    depth_filtered_pubs_[idx], depth_filtered_pubs_be_[idx],
    toMsg(filtered, slot.stamp, world), "depth_filtered");
  publishCloud(
    depth_deviate_pubs_[idx], depth_deviate_pubs_be_[idx],
    toMsg(deviate, slot.stamp, world), "depth_deviate");
}

void WallPlaneFilterNode::reportDepthTf()
{
  std::ostringstream ss;
  bool any = false;
  {
    std::lock_guard<std::mutex> lock(depth_tf_mutex_);
    const auto now = std::chrono::steady_clock::now();
    const double period = std::max(0.5, params_.depth_wall_tf_report_sec);
    if (std::chrono::duration<double>(now - depth_tf_report_last_).count() < period) {
      return;
    }
    depth_tf_report_last_ = now;
    for (size_t i = 0; i < depth_tf_total_.size(); ++i) {
      if (depth_tf_total_[i] == 0) {
        continue;
      }
      any = true;
      ss << ' ' << depth_cam_names_[i] << '=' << depth_tf_fail_[i] << '/' << depth_tf_total_[i] <<
        " (" << std::fixed << std::setprecision(1) <<
        100.0 * static_cast<double>(depth_tf_fail_[i]) /
        static_cast<double>(depth_tf_total_[i]) << "%)";
      depth_tf_total_[i] = 0;
      depth_tf_fail_[i] = 0;
    }
  }
  if (any) {
    RCLCPP_INFO(get_logger(), "depth TF fail rate:%s", ss.str().c_str());
  }
}

void WallPlaneFilterNode::processDepthWalls(
  size_t idx, const CloudT & wall_world, const Eigen::Vector2d & robot_xy,
  const rclcpp::Time & stamp)
{
  if (idx >= depth_cam_tracks_.size()) {
    return;
  }
  HoughConfig hough_cfg = params_.depth_hough;
  {
    std::lock_guard<std::mutex> lock(depth_wall_mutex_);
    holdDepthGrid(idx, robot_xy, hough_cfg);
  }
  const auto result = extractHoughWalls(wall_world, robot_xy, hough_cfg);
  if (params_.depth_hough.debug) {
    publishHoughDebug(result, stamp, wallWorkFrame(), true);
  }

  const WallTrackConfig track_cfg = trackConfigFrom(params_.depth_hough, params_.wall_track);
  std::vector<WallLine> cam_tracked;
  std::vector<WallLine> all_tracked;
  {
    std::lock_guard<std::mutex> lock(depth_wall_mutex_);
    cam_tracked = depth_cam_tracks_[idx].update(result.lines, stamp.seconds(), track_cfg);
    // Age sibling camera tracks so lost walls expire without waiting for fusion.
    for (size_t i = 0; i < depth_cam_tracks_.size(); ++i) {
      if (i == idx) {
        continue;
      }
      depth_cam_tracks_[i].update({}, stamp.seconds(), track_cfg);
    }
    all_tracked = collectDepthTrackedLines(stamp.seconds());
    validated_depth_lines_ = all_tracked;
    validated_depth_stamp_ = stamp;
  }

  publishDepthCamPlanes(idx, cam_tracked, stamp, wallWorkFrame());
  publishTrackedMarkers(all_tracked, result.corners, stamp, wallWorkFrame(), true);
}

std::vector<WallLine> WallPlaneFilterNode::collectDepthTrackedLines(double /*stamp_sec*/)
{
  std::vector<WallLine> all;
  for (const auto & tracker : depth_cam_tracks_) {
    const auto lines = tracker.confirmed();
    all.insert(all.end(), lines.begin(), lines.end());
  }
  return fuseWallLines(
    all, params_.depth_hough.merge_angle_deg, params_.depth_hough.merge_distance,
    params_.depth_hough.merge_gap, params_.wall_fuse_min_support_points);
}

void WallPlaneFilterNode::publishDepthCamPlanes(
  size_t idx, const std::vector<WallLine> & lines, const rclcpp::Time & stamp,
  const std::string & frame)
{
  if (idx >= depth_cam_plane_pubs_.size()) {
    return;
  }
  const std::string output = outputFrame();
  wall_plane_filter::msg::WallPlaneArray planes;
  planes.header.frame_id = output;
  planes.header.stamp = stamp;
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  if (!(output == frame || lookupIso(output, frame, stamp, transform))) {
    depth_cam_plane_pubs_[idx]->publish(planes);
    return;
  }
  for (const auto & line_in : lines) {
    WallLine line = line_in;
    if (output != frame) {
      line = transformLine(line_in, transform);
    }
    planes.planes.push_back(
      toWallPlaneMsg(line, true, params_.depth_hough.z_min, params_.depth_hough.z_max));
  }
  depth_cam_plane_pubs_[idx]->publish(planes);
}

void WallPlaneFilterNode::holdDepthGrid(
  size_t idx, const Eigen::Vector2d & robot, HoughConfig & config)
{
  if (!params_.depth_hough.follow_sensor) {
    return;
  }
  if (idx >= depth_grid_origins_.size()) {
    return;
  }
  auto & origin = depth_grid_origins_[idx];
  char & valid = depth_grid_valid_[idx];
  const double res = config.resolution;
  const int side = static_cast<int>(std::ceil(2.0 * config.half_extent / res));
  const double span = side * res;
  const double margin = std::max(0.5, 2.0 * res);
  const bool inside = valid != 0 &&
    robot.x() > origin.x() + margin &&
    robot.y() > origin.y() + margin &&
    robot.x() < origin.x() + span - margin &&
    robot.y() < origin.y() + span - margin;
  if (!inside) {
    origin = (robot / res).array().floor().matrix() * res -
      Eigen::Vector2d::Constant(span * 0.5);
    valid = 1;
  }
  config.follow_sensor = false;
  config.origin_x = origin.x();
  config.origin_y = origin.y();
}

wall_plane_filter::msg::WallPlane WallPlaneFilterNode::toWallPlaneMsg(
  const WallLine & line, bool depth, double fallback_z_min, double fallback_z_max) const
{
  wall_plane_filter::msg::WallPlane msg;
  double z_min = line.z_min;
  double z_max = line.z_max;
  if (!std::isfinite(z_min) || !std::isfinite(z_max) || z_max <= z_min) {
    z_min = fallback_z_min;
    z_max = fallback_z_max;
  }
  msg.normal.x = line.normal.x();
  msg.normal.y = line.normal.y();
  msg.normal.z = 0.0;
  msg.rho = line.rho;
  msg.a.x = line.a.x();
  msg.a.y = line.a.y();
  msg.a.z = z_min;
  msg.b.x = line.b.x();
  msg.b.y = line.b.y();
  msg.b.z = z_min;
  msg.z_min = z_min;
  msg.z_max = z_max;
  msg.corners[0] = pointMsg(Eigen::Vector3d(line.a.x(), line.a.y(), z_min));
  msg.corners[1] = pointMsg(Eigen::Vector3d(line.b.x(), line.b.y(), z_min));
  msg.corners[2] = pointMsg(Eigen::Vector3d(line.b.x(), line.b.y(), z_max));
  msg.corners[3] = pointMsg(Eigen::Vector3d(line.a.x(), line.a.y(), z_max));
  msg.length = (line.b - line.a).norm();
  msg.height = z_max - z_min;
  msg.sigma = line.sigma;
  msg.support_ratio = line.support_ratio;
  msg.support_points = static_cast<uint32_t>(line.support_points);
  msg.source = depth ? wall_plane_filter::msg::WallPlane::SOURCE_DEPTH
                     : wall_plane_filter::msg::WallPlane::SOURCE_LIDAR;
  return msg;
}

void WallPlaneFilterNode::updateElevatorDoor(const CloudT & cloud_base, const rclcpp::Time & stamp)
{
  if (!params_.elevator_door_enable || !elevator_door_pub_) {
    return;
  }
  using Status = wall_plane_filter::msg::ElevatorDoorStatus;
  Status msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = params_.target_frame;
  msg.state = Status::UNKNOWN;
  msg.gap_m = 0.0;
  msg.door_range_m = std::numeric_limits<double>::quiet_NaN();
  msg.open_ratio = 0.0;
  msg.confidence = 0.0;
  msg.valid_bins = 0;
  msg.open_bins = 0;
  msg.near_bins = 0;

  const int bins = std::max(3, params_.elevator_door_bins);
  const double half_w = std::max(0.1, params_.elevator_door_half_width);
  const double x_min = params_.elevator_door_x_min;
  const double x_max = std::max(x_min + 0.1, params_.elevator_door_x_max);
  const double z_min = params_.elevator_door_z_min;
  const double z_max = params_.elevator_door_z_max;
  const double bin_w = (2.0 * half_w) / static_cast<double>(bins);

  std::vector<std::vector<double>> xs(static_cast<size_t>(bins));
  for (const auto & p : cloud_base.points) {
    if (p.x < x_min || p.x > x_max || p.z < z_min || p.z > z_max) {
      continue;
    }
    if (std::abs(p.y) > half_w) {
      continue;
    }
    int b = static_cast<int>(std::floor((p.y + half_w) / bin_w));
    b = std::clamp(b, 0, bins - 1);
    xs[static_cast<size_t>(b)].push_back(p.x);
  }

  std::vector<double> med(static_cast<size_t>(bins), std::numeric_limits<double>::quiet_NaN());
  uint32_t valid = 0;
  for (int i = 0; i < bins; ++i) {
    auto & v = xs[static_cast<size_t>(i)];
    if (static_cast<int>(v.size()) < params_.elevator_door_min_points_bin) {
      continue;
    }
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    med[static_cast<size_t>(i)] = v[mid];
    ++valid;
  }
  msg.valid_bins = valid;
  if (static_cast<int>(valid) < params_.elevator_door_min_valid_bins) {
    std::lock_guard<std::mutex> lock(elevator_door_mutex_);
    msg.state = elevator_door_state_;
    elevator_door_pub_->publish(msg);
    return;
  }

  // Near plane ≈ lower quartile of valid bin medians (door panel / sill distance).
  std::vector<double> near_candidates;
  near_candidates.reserve(valid);
  for (double m : med) {
    if (std::isfinite(m)) {
      near_candidates.push_back(m);
    }
  }
  std::sort(near_candidates.begin(), near_candidates.end());
  const double near_x = near_candidates[near_candidates.size() / 4];
  msg.door_range_m = near_x;
  const double open_x = near_x + std::max(0.1, params_.elevator_door_open_margin);

  uint32_t open_bins = 0;
  uint32_t near_bins = 0;
  double gap_sum = 0.0;
  for (double m : med) {
    if (!std::isfinite(m)) {
      continue;
    }
    if (m >= open_x) {
      ++open_bins;
      gap_sum += (m - near_x);
    } else {
      ++near_bins;
    }
  }
  msg.open_bins = open_bins;
  msg.near_bins = near_bins;
  msg.open_ratio = static_cast<double>(open_bins) / static_cast<double>(valid);
  msg.gap_m = open_bins > 0 ? (gap_sum / static_cast<double>(open_bins)) : 0.0;
  msg.confidence = std::min(1.0, static_cast<double>(valid) / static_cast<double>(bins));

  const uint8_t measured = (msg.open_ratio >= params_.elevator_door_open_ratio) ?
    Status::OPEN : Status::CLOSED;

  {
    std::lock_guard<std::mutex> lock(elevator_door_mutex_);
    if (measured == elevator_door_pending_) {
      ++elevator_door_pending_hits_;
    } else {
      elevator_door_pending_ = measured;
      elevator_door_pending_hits_ = 1;
    }
    if (elevator_door_pending_hits_ >= std::max(1, params_.elevator_door_vote_frames)) {
      elevator_door_state_ = elevator_door_pending_;
    }
    msg.state = elevator_door_state_;
  }

  elevator_door_pub_->publish(msg);
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 1000,
    "elevator door state=%u open_ratio=%.2f near_x=%.2f gap=%.2f valid=%u/%d",
    static_cast<unsigned>(msg.state), msg.open_ratio, near_x, msg.gap_m, valid, bins);
}

void WallPlaneFilterNode::publishTrackedMarkers(
  const std::vector<WallLine> & lines, const std::vector<Eigen::Vector2d> & corners,
  const rclcpp::Time & stamp, const std::string & frame, bool depth)
{
  using Marker = visualization_msgs::msg::Marker;
  const std::string output = outputFrame();
  const auto marker_pub = depth ? depth_wall_marker_pub_ : lidar_wall_marker_pub_;
  const auto plane_pub = depth ? depth_wall_plane_pub_ : lidar_wall_plane_pub_;
  const double fallback_z_min = depth ? params_.depth_hough.z_min : params_.lidar_hough.z_min;
  const double fallback_z_max = depth ? params_.depth_hough.z_max : params_.lidar_hough.z_max;

  wall_plane_filter::msg::WallPlaneArray planes;
  planes.header.frame_id = output;
  planes.header.stamp = stamp;

  // Empty observation: republish last markers with refreshed stamp/lifetime (coasting).
  if (lines.empty()) {
    visualization_msgs::msg::MarkerArray coast;
    wall_plane_filter::msg::WallPlaneArray coast_planes;
    bool have_coast = false;
    {
      std::lock_guard<std::mutex> lock(depth_wall_mutex_);
      auto & last_markers = depth ? last_depth_wall_markers_ : last_lidar_wall_markers_;
      auto & last_planes = depth ? last_depth_wall_planes_ : last_lidar_wall_planes_;
      if (!last_markers.markers.empty()) {
        coast = last_markers;
        for (auto & m : coast.markers) {
          m.header.stamp = stamp;
          if (m.action == Marker::ADD) {
            m.lifetime = rclcpp::Duration::from_seconds(params_.wall_marker_lifetime);
          }
        }
        last_planes.header.stamp = stamp;
        coast_planes = last_planes;
        have_coast = true;
      }
    }
    if (have_coast) {
      marker_pub->publish(coast);
      plane_pub->publish(coast_planes);
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "wall planes %s empty; coasting last markers=%zu",
        depth ? "depth" : "lidar", coast.markers.size());
    } else {
      plane_pub->publish(planes);
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "wall planes %s empty; no previous markers to coast", depth ? "depth" : "lidar");
    }
    return;
  }

  visualization_msgs::msg::MarkerArray final;
  Marker clear;
  clear.header = planes.header;
  clear.action = Marker::DELETEALL;
  final.markers.push_back(clear);

  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  const bool tf_ok = output == frame || lookupIso(output, frame, stamp, transform);
  if (tf_ok) {
    Marker edges;
    edges.header = clear.header;
    edges.ns = depth ? "depth_walls" : "lidar_walls";
    edges.id = 0;
    edges.type = Marker::LINE_LIST;
    edges.action = Marker::ADD;
    edges.pose.orientation.w = 1.0;
    edges.scale.x = params_.wall_line_width;
    edges.color.r = depth ? 0.2f : 1.0f;
    edges.color.g = depth ? 1.0f : 0.55f;
    edges.color.b = depth ? 0.9f : 0.1f;
    edges.color.a = 1.0f;
    edges.lifetime = rclcpp::Duration::from_seconds(params_.wall_marker_lifetime);

    Marker faces = edges;
    faces.ns = depth ? "depth_wall_planes" : "lidar_wall_planes";
    faces.id = 1;
    faces.type = Marker::TRIANGLE_LIST;
    faces.scale.x = faces.scale.y = faces.scale.z = 1.0;
    faces.color.a = 0.35f;

    for (const auto & line_in : lines) {
      WallLine line = line_in;
      if (output != frame) {
        line = transformLine(line_in, transform);
      }
      planes.planes.push_back(toWallPlaneMsg(line, depth, fallback_z_min, fallback_z_max));
      const auto & plane = planes.planes.back();
      // Bottom and top edges for the old line view.
      edges.points.push_back(plane.corners[0]);
      edges.points.push_back(plane.corners[1]);
      edges.points.push_back(plane.corners[3]);
      edges.points.push_back(plane.corners[2]);
      // Two triangles: 0-1-2 and 0-2-3.
      faces.points.push_back(plane.corners[0]);
      faces.points.push_back(plane.corners[1]);
      faces.points.push_back(plane.corners[2]);
      faces.points.push_back(plane.corners[0]);
      faces.points.push_back(plane.corners[2]);
      faces.points.push_back(plane.corners[3]);
    }
    final.markers.push_back(edges);
    final.markers.push_back(faces);
    if (!corners.empty()) {
      Marker corner_marker = edges;
      corner_marker.ns += "_corners";
      corner_marker.id = 2;
      corner_marker.type = Marker::POINTS;
      corner_marker.scale.x = corner_marker.scale.y = 2 * params_.wall_line_width;
      corner_marker.points.clear();
      for (const auto & p : corners) {
        const Eigen::Vector3d q = transform * Eigen::Vector3d(p.x(), p.y(), params_.wall_display_z);
        corner_marker.points.push_back(pointMsg(q));
      }
      final.markers.push_back(corner_marker);
    }
  }
  marker_pub->publish(final);
  plane_pub->publish(planes);
  {
    std::lock_guard<std::mutex> lock(depth_wall_mutex_);
    if (depth) {
      last_depth_wall_markers_ = final;
      last_depth_wall_planes_ = planes;
    } else {
      last_lidar_wall_markers_ = final;
      last_lidar_wall_planes_ = planes;
    }
  }
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 2000,
    "wall planes %s published=%zu output_tf=%d",
    depth ? "depth" : "lidar", planes.planes.size(), tf_ok ? 1 : 0);
}

void WallPlaneFilterNode::publishHoughDebug(
  const HoughResult & result, const rclcpp::Time & stamp,
  const std::string & frame, bool depth)
{
  using Marker = visualization_msgs::msg::Marker;
  const auto grid_pub = depth ? depth_grid_pub_ : lidar_grid_pub_;
  const auto debug_pub = depth ? depth_debug_pub_ : lidar_debug_pub_;
  if (grid_pub && debug_pub) {
    nav_msgs::msg::OccupancyGrid grid;
    grid.header.frame_id = frame;
    grid.header.stamp = stamp;
    grid.info.resolution = result.grid.resolution;
    grid.info.width = result.grid.width;
    grid.info.height = result.grid.height;
    grid.info.origin.position.x = result.grid.origin.x();
    grid.info.origin.position.y = result.grid.origin.y();
    grid.info.origin.orientation.w = 1.0;
    grid.data.reserve(result.grid.binary.size());
    for (auto pixel : result.grid.binary) {grid.data.push_back(pixel ? 100 : 0);}
    grid_pub->publish(grid);
    visualization_msgs::msg::MarkerArray debug;
    Marker clear;
    clear.header = grid.header;
    clear.action = Marker::DELETEALL;
    debug.markers.push_back(clear);
    auto add_lines = [&](const std::vector<WallLine> & source, const std::string & name,
        float r, float g, float b, double z) {
        Marker marker;
        marker.header = grid.header;
        marker.ns = name;
        marker.type = Marker::LINE_LIST;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = params_.wall_line_width;
        marker.color.r = r; marker.color.g = g; marker.color.b = b; marker.color.a = 1;
        marker.lifetime = rclcpp::Duration::from_seconds(params_.wall_marker_lifetime);
        for (const auto & l : source) {
          marker.points.push_back(pointMsg(Eigen::Vector3d(l.a.x(), l.a.y(), z)));
          marker.points.push_back(pointMsg(Eigen::Vector3d(l.b.x(), l.b.y(), z)));
        }
        debug.markers.push_back(marker);
      };
    add_lines(result.hough_lines, "hough", 0.2f, 0.4f, 1.0f, 0.03);
    add_lines(result.candidates, "candidates", 1.0f, 0.6f, 0.1f, 0.06);
    add_lines(result.lines, "final", 0.1f, 1.0f, 0.2f, 0.09);
    Marker points;
    points.header = grid.header;
    points.ns = "candidate_intersections";
    points.type = Marker::POINTS;
    points.pose.orientation.w = 1.0;
    points.scale.x = points.scale.y = 0.08;
    points.color.r = points.color.b = points.color.a = 1;
    points.lifetime = rclcpp::Duration::from_seconds(params_.wall_marker_lifetime);
    for (const auto & p : result.intersections) {points.points.push_back(pointMsg(Eigen::Vector3d(p.x(), p.y(), 0.12)));}
    debug.markers.push_back(points);
    for (size_t i = 0; i < result.lines.size(); ++i) {
      const auto & l = result.lines[i];
      Marker label;
      label.header = grid.header;
      label.ns = "final_info"; label.id = static_cast<int>(i);
      label.type = Marker::TEXT_VIEW_FACING;
      label.pose.orientation.w = 1.0;
      label.pose.position = pointMsg(Eigen::Vector3d((l.a.x() + l.b.x()) / 2, (l.a.y() + l.b.y()) / 2, 0.3));
      label.scale.z = 0.10;
      label.color.r = label.color.g = label.color.b = label.color.a = 1;
      label.lifetime = points.lifetime;
      std::ostringstream text;
      text << std::fixed << std::setprecision(2) << l.normal.x() << "x+" << l.normal.y() << "y=" << l.rho
           << " n=" << l.support_points << " support=" << l.support_ratio
           << " len=" << (l.b - l.a).norm() << " theta="
           << std::atan2(l.b.y() - l.a.y(), l.b.x() - l.a.x()) * 180.0 / std::acos(-1.0)
           << " z=[" << l.z_min << "," << l.z_max << "] sigma=" << l.sigma;
      label.text = text.str();
      debug.markers.push_back(label);
    }
    debug_pub->publish(debug);
  }
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 2000,
    "Hough %s input=%zu projected=%zu cells=%zu hough=%zu candidates=%zu final=%zu corners=%zu ms=%.2f",
    depth ? "depth" : "lidar", result.input_points, result.projected_points, result.occupied_cells,
    result.hough_lines.size(), result.candidates.size(), result.lines.size(), result.corners.size(),
    result.elapsed_ms);
}

}  // namespace wall_plane_filter
