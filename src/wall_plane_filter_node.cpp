#include "wall_plane_filter/wall_plane_filter_node.hpp"

#include <tf2_eigen/tf2_eigen.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

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

void keepInRoi(
  const CloudT & in, const Eigen::Isometry3d & T_bc, const Params & params, CloudT & out)
{
  out.clear();
  out.reserve(in.size());
  for (const auto & p : in.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {continue;}
    const Eigen::Vector3d q = T_bc * Eigen::Vector3d(p.x, p.y, p.z);
    if (q.z() < params.depth_z_min || q.z() > params.depth_z_max) {continue;}
    const double r = std::hypot(q.x(), q.y());
    if (r < params.depth_range_min || r > params.depth_range_max) {continue;}
    out.push_back(pcl::PointXYZ(
      static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z())));
  }
  finishCloud(out);
}

void voxelDownsample(const CloudT & in, double leaf, CloudT & out)
{
  out.clear();
  if (in.empty() || !(leaf > 1e-9)) {
    out = in;
    finishCloud(out);
    return;
  }
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(in.makeShared());
  const float leaf_f = static_cast<float>(leaf);
  vg.setLeafSize(leaf_f, leaf_f, leaf_f);
  vg.filter(out);
  finishCloud(out);
}

bool bilateralSmoothXyz(const CloudT & in, CloudT & out, double sigma_s, double sigma_r)
{
  out.clear();
  if (in.empty() || !(sigma_s > 1e-9) || !(sigma_r > 1e-9)) {
    out = in; finishCloud(out); return false;
  }
  const double radius = 3.0 * sigma_s;
  const double inv_2ss = 1.0 / (2.0 * sigma_s * sigma_s);
  const double inv_2sr = 1.0 / (2.0 * sigma_r * sigma_r);
  pcl::KdTreeFLANN<pcl::PointXYZ> tree;
  auto input = in.makeShared();
  tree.setInputCloud(input);
  out.resize(in.size());
  std::vector<int> ids;
  std::vector<float> dist2;
  for (size_t i = 0; i < in.size(); ++i) {
    const auto & p = in.points[i];
    ids.clear(); dist2.clear();
    if (tree.radiusSearch(p, radius, ids, dist2) < 1) {
      out.points[i] = p; continue;
    }
    double wsum = 0, sx = 0, sy = 0, sz = 0;
    for (size_t k = 0; k < ids.size(); ++k) {
      const auto & q = in.points[static_cast<size_t>(ids[k])];
      const double dxy2 = static_cast<double>(dist2[k]);
      const double dz = static_cast<double>(q.z) - static_cast<double>(p.z);
      const double w = std::exp(-dxy2 * inv_2ss - (dz * dz) * inv_2sr);
      wsum += w; sx += w * q.x; sy += w * q.y; sz += w * q.z;
    }
    if (!(wsum > 1e-12)) {out.points[i] = p; continue;}
    out.points[i].x = static_cast<float>(sx / wsum);
    out.points[i].y = static_cast<float>(sy / wsum);
    out.points[i].z = static_cast<float>(sz / wsum);
  }
  finishCloud(out);
  return true;
}

struct PrepStats {
  size_t crop{0}, voxel{0}, ror_drop{0}, sor_drop{0}, out{0}; bool bilateral{false};
};

void appendRemoved(
  const CloudT & src, const pcl::IndicesConstPtr & removed, CloudT & deviate, size_t & drop_count)
{
  if (!removed || removed->empty()) {return;}
  drop_count += removed->size();
  for (const auto i : *removed) {
    if (i < 0 || static_cast<size_t>(i) >= src.size()) {continue;}
    deviate.push_back(src.points[static_cast<size_t>(i)]);
  }
  finishCloud(deviate);
}

void preprocessDepthCloud(
  const CloudT & cam_cloud, const Eigen::Isometry3d & T_bc, const Params & params,
  CloudT & filtered, CloudT & deviate, PrepStats & st)
{
  st = PrepStats{};
  filtered.clear(); deviate.clear();
  CloudT crop;
  keepInRoi(cam_cloud, T_bc, params, crop);
  st.crop = crop.size();
  if (crop.empty()) {finishCloud(filtered); finishCloud(deviate); return;}

  CloudT voxel;
  voxelDownsample(crop, params.depth_voxel_xy, voxel);
  st.voxel = voxel.size();
  if (voxel.empty()) {finishCloud(filtered); finishCloud(deviate); return;}

  // Optional ROR coarse reject when the cloud is dense enough, then SOR refine.
  CloudT after_ror = voxel;
  if (params.depth_ror_enable &&
    voxel.size() >= static_cast<size_t>(std::max(1, params.depth_ror_min_points)))
  {
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror(true);
    ror.setInputCloud(voxel.makeShared());
    ror.setRadiusSearch(std::max(1e-4, params.depth_ror_radius));
    ror.setMinNeighborsInRadius(std::max(1, params.depth_ror_min_neighbors));
    ror.filter(after_ror);
    appendRemoved(voxel, ror.getRemovedIndices(), deviate, st.ror_drop);
  }
  finishCloud(after_ror);

  CloudT after_sor = after_ror;
  if (params.depth_sor_enable &&
    after_ror.size() >= static_cast<size_t>(std::max(3, params.depth_sor_mean_k)))
  {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor(true);
    sor.setInputCloud(after_ror.makeShared());
    sor.setMeanK(std::max(3, params.depth_sor_mean_k));
    sor.setStddevMulThresh(std::max(0.1, params.depth_sor_stddev_mul));
    sor.filter(after_sor);
    appendRemoved(after_ror, sor.getRemovedIndices(), deviate, st.sor_drop);
  }
  finishCloud(after_sor);

  if (params.depth_bilateral_enable && !after_sor.empty()) {
    st.bilateral = bilateralSmoothXyz(
      after_sor, filtered, params.depth_bilateral_sigma_s, params.depth_bilateral_sigma_r);
    if (!st.bilateral) {filtered = std::move(after_sor);}
  } else {
    filtered = std::move(after_sor);
  }
  finishCloud(filtered);
  st.out = filtered.size();
}

bool nearestWallNormal(
  const Eigen::Vector2d & xy, const std::vector<WallLine> & walls, double end_margin,
  double & abs_d, double & signed_d, Eigen::Vector2d & normal_out)
{
  abs_d = std::numeric_limits<double>::infinity();
  signed_d = 0.0;
  normal_out = Eigen::Vector2d::Zero();
  bool found = false;
  const double margin = std::max(0.0, end_margin);
  for (const auto & wall : walls) {
    Eigen::Vector2d n = wall.normal;
    const double nn = n.norm();
    if (nn < 1e-9) {continue;}
    n /= nn;
    const double sd = n.dot(xy) - n.dot(wall.a);
    const double ad = std::abs(sd);
    Eigen::Vector2d dir = wall.b - wall.a;
    const double len = dir.norm();
    if (len < 1e-9) {continue;}
    dir /= len;
    const double t = dir.dot(xy - wall.a);
    const double t_clamped = std::clamp(t, 0.0, len);
    const Eigen::Vector2d foot = wall.a + dir * t_clamped;
    const double d_foot = (xy - foot).norm();
    if (t >= 0.0 && t <= len) {
      if (ad < abs_d) {
        abs_d = ad; signed_d = sd; normal_out = n; found = true;
      }
      continue;
    }
    if (d_foot <= margin) {
      if (d_foot < abs_d) {
        abs_d = d_foot; signed_d = sd; normal_out = n; found = true;
      }
      continue;
    }
    if (d_foot < abs_d) {
      abs_d = d_foot; signed_d = sd; normal_out = Eigen::Vector2d::Zero(); found = true;
    }
  }
  return found;
}

size_t cullWallBand(
  CloudT & cloud, CloudT & deviate, const std::vector<WallLine> & walls,
  double keep, double outer, bool project, const Eigen::Vector2d & robot_xy,
  double noise_k, double noise_max, const Eigen::Vector2d * sensor_xy, size_t * projected_out)
{
  if (projected_out) {*projected_out = 0;}
  if (cloud.empty() || walls.empty() || !(keep > 0.0) || !(outer > keep)) {return 0;}
  const Eigen::Vector2d noise_origin = sensor_xy ? *sensor_xy : robot_xy;
  const double end_margin = std::max({keep, outer * 0.5, 0.12});
  CloudT kept; kept.reserve(cloud.size());
  size_t dropped = 0, projected = 0;
  for (const auto & p : cloud.points) {
    const Eigen::Vector2d xy(p.x, p.y);
    double abs_d = 0, signed_d = 0;
    Eigen::Vector2d n = Eigen::Vector2d::Zero();
    if (!nearestWallNormal(xy, walls, end_margin, abs_d, signed_d, n)) {
      kept.push_back(p); continue;
    }
    const bool on = n.squaredNorm() > 0.5;
    const double range = (xy - noise_origin).norm();
    const double noise = std::min(noise_max, noise_k * range * range);
    const double keep_pt = std::max(keep, noise);
    const double outer_pt = std::max(outer, keep_pt + 0.02);
    if (on && abs_d > keep_pt && abs_d <= outer_pt) {
      deviate.push_back(p); ++dropped; continue;
    }
    if (on && abs_d <= keep_pt && project) {
      pcl::PointXYZ q = p;
      q.x = static_cast<float>(p.x - signed_d * n.x());
      q.y = static_cast<float>(p.y - signed_d * n.y());
      kept.push_back(q); ++projected; continue;
    }
    kept.push_back(p);
  }
  cloud = std::move(kept);
  finishCloud(cloud); finishCloud(deviate);
  if (projected_out) {*projected_out = projected;}
  return dropped;
}

bool tryFuseWallLine(
  WallLine & a, const WallLine & b, double merge_angle_deg, double merge_distance, double merge_gap)
{
  const Eigen::Vector2d da = a.b - a.a;
  const double la = da.norm();
  if (la < 1e-9) {return false;}
  const Eigen::Vector2d d = da / la;
  const Eigen::Vector2d db = b.b - b.a;
  const double lb = db.norm();
  if (lb < 1e-9) {return false;}
  if (std::abs(d.dot(db / lb)) < std::cos(merge_angle_deg * M_PI / 180.0)) {return false;}
  if (std::abs(a.normal.dot(b.a) - a.rho) > merge_distance ||
    std::abs(a.normal.dot(b.b) - a.rho) > merge_distance)
  {
    return false;
  }
  const double t0 = d.dot(b.a - a.a), t1 = d.dot(b.b - a.a);
  const double lo = std::min(t0, t1), hi = std::max(t0, t1);
  if (lo > la + merge_gap || hi < -merge_gap) {return false;}
  const double nlo = std::min(0.0, lo), nhi = std::max(la, hi);
  const Eigen::Vector2d na = a.a + d * nlo, nb = a.a + d * nhi;
  a.a = na; a.b = nb;
  a.support_points = std::max(a.support_points, b.support_points);
  return true;
}

struct ResidualStats
{
  size_t count{0};
  double min_e{0}, max_e{0}, mean_e{0}, rms{0};
  double min_abs{0}, max_abs{0}, mean_abs{0}, std_abs{0};
};

ResidualStats accumulateResiduals(const std::vector<double> & errs)
{
  ResidualStats s;
  if (errs.empty()) {return s;}
  s.count = errs.size();
  s.min_e = s.max_e = errs.front();
  double sum = 0, sum_sq = 0, sum_abs = 0, sum_abs_sq = 0;
  s.min_abs = s.max_abs = std::abs(errs.front());
  for (double e : errs) {
    const double ae = std::abs(e);
    s.min_e = std::min(s.min_e, e);
    s.max_e = std::max(s.max_e, e);
    s.min_abs = std::min(s.min_abs, ae);
    s.max_abs = std::max(s.max_abs, ae);
    sum += e;
    sum_sq += e * e;
    sum_abs += ae;
    sum_abs_sq += ae * ae;
  }
  const double n = static_cast<double>(s.count);
  s.mean_e = sum / n;
  s.rms = std::sqrt(sum_sq / n);
  s.mean_abs = sum_abs / n;
  const double var = std::max(0.0, sum_abs_sq / n - s.mean_abs * s.mean_abs);
  s.std_abs = std::sqrt(var);
  return s;
}

/** Support-point normal residuals e = n·xy − ρ within support_band on each wall segment. */
void collectWallSupportResiduals(
  const std::vector<WallLine> & lines, const CloudT & cloud, double support_band,
  std::vector<ResidualStats> & per_wall, ResidualStats & global, std::vector<double> & all_abs)
{
  per_wall.assign(lines.size(), ResidualStats{});
  global = ResidualStats{};
  all_abs.clear();
  if (lines.empty() || cloud.empty() || !(support_band > 0.0)) {return;}
  std::vector<double> all_signed;
  all_signed.reserve(cloud.size());
  for (size_t wi = 0; wi < lines.size(); ++wi) {
    const WallLine & line = lines[wi];
    const double len = (line.b - line.a).norm();
    if (len < 1e-9) {continue;}
    Eigen::Vector2d d = (line.b - line.a) / len;
    Eigen::Vector2d n = line.normal;
    const double nn = n.norm();
    if (nn > 1e-9) {n /= nn;} else {n = Eigen::Vector2d(-d.y(), d.x());}
    const double rho = n.dot(line.a);
    std::vector<double> errs;
    errs.reserve(64);
    for (const auto & p : cloud.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {continue;}
      const Eigen::Vector2d xy(p.x, p.y);
      const double e = n.dot(xy) - rho;
      if (std::abs(e) > support_band) {continue;}
      const double t = d.dot(xy - line.a);
      if (t < -1e-6 || t > len + 1e-6) {continue;}
      errs.push_back(e);
      all_signed.push_back(e);
      all_abs.push_back(std::abs(e));
    }
    per_wall[wi] = accumulateResiduals(errs);
  }
  global = accumulateResiduals(all_signed);
}

void finalizeWallGeometry(
  std::vector<WallLine> & lines, const CloudT * support_cloud, const HoughConfig & hough,
  const Params & params)
{
  if (params.wall_clamp_to_support && support_cloud && !support_cloud->empty()) {
    clampWallsToSupportCloud(
      lines, *support_cloud, hough.support_band, params.wall_extent_margin_m,
      hough.min_line_length);
  }
  if (params.corner_snap_enable) {
    const double snap = params.corner_snap_dist > 1e-9 ?
      params.corner_snap_dist : hough.intersection_extension;
    snapWallCorners(lines, hough.parallel_angle_deg, snap, hough.min_line_length);
  }
}

std::vector<WallLine> fuseWallLines(
  const std::vector<WallLine> & lines, double merge_angle_deg, double merge_distance,
  double merge_gap, int min_support)
{
  std::vector<WallLine> fused;
  for (const auto & line : lines) {
    if (static_cast<int>(line.support_points) < min_support && min_support > 0) {
      // Still allow fuse attempt; filter after.
    }
    bool ok = false;
    for (auto & prev : fused) {
      if (tryFuseWallLine(prev, line, merge_angle_deg, merge_distance, merge_gap)) {
        ok = true; break;
      }
    }
    if (!ok) {fused.push_back(line);}
  }
  if (min_support > 0) {
    std::vector<WallLine> kept;
    for (const auto & l : fused) {
      if (static_cast<int>(l.support_points) >= min_support || l.support_points == 0) {
        kept.push_back(l);
      }
    }
    // If support_points often unset after fuse, keep all fused.
    if (kept.empty() && !fused.empty()) {return fused;}
    return kept.empty() ? fused : kept;
  }
  return fused;
}

WallLine transformLine(const WallLine & line, const Eigen::Isometry3d & tf)
{
  WallLine out = line;
  const Eigen::Vector3d a = tf * Eigen::Vector3d(line.a.x(), line.a.y(), 0.0);
  const Eigen::Vector3d b = tf * Eigen::Vector3d(line.b.x(), line.b.y(), 0.0);
  out.a = a.head<2>();
  out.b = b.head<2>();
  const Eigen::Vector2d d = out.b - out.a;
  if (d.norm() < 1e-9) {return out;}
  const Eigen::Vector2d dir = d.normalized();
  out.normal = Eigen::Vector2d(-dir.y(), dir.x());
  if (out.normal.dot(line.normal) < 0.0) {out.normal = -out.normal;}
  out.rho = out.normal.dot(out.a);
  return out;
}

geometry_msgs::msg::Point pointMsg(const Eigen::Vector3d & p)
{
  geometry_msgs::msg::Point g;
  g.x = p.x(); g.y = p.y(); g.z = p.z();
  return g;
}

wall_plane_filter::msg::WallPlane toWallPlaneMsg(
  const WallLine & line, double z_min, double z_max)
{
  wall_plane_filter::msg::WallPlane msg;
  msg.normal.x = line.normal.x();
  msg.normal.y = line.normal.y();
  msg.normal.z = 0.0;
  msg.rho = static_cast<float>(line.rho);
  msg.z_min = static_cast<float>(std::isfinite(line.z_min) ? line.z_min : z_min);
  msg.z_max = static_cast<float>(std::isfinite(line.z_max) ? line.z_max : z_max);
  msg.a.x = line.a.x(); msg.a.y = line.a.y(); msg.a.z = msg.z_min;
  msg.b.x = line.b.x(); msg.b.y = line.b.y(); msg.b.z = msg.z_min;
  const float z0 = msg.z_min, z1 = msg.z_max;
  msg.corners[0] = pointMsg({line.a.x(), line.a.y(), z0});
  msg.corners[1] = pointMsg({line.b.x(), line.b.y(), z0});
  msg.corners[2] = pointMsg({line.b.x(), line.b.y(), z1});
  msg.corners[3] = pointMsg({line.a.x(), line.a.y(), z1});
  msg.support_points = static_cast<uint32_t>(line.support_points);
  msg.support_ratio = static_cast<float>(line.support_ratio);
  msg.sigma = static_cast<float>(line.sigma);
  return msg;
}

}  // namespace

WallPlaneFilterNode::WallPlaneFilterNode(const rclcpp::NodeOptions & options)
: Node("wall_plane_filter", options)
{
  loadParams();
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  lidar_wall_plane_pub_ = create_publisher<wall_plane_filter::msg::WallPlaneArray>(
    "/wall_plane_filter/lidar_wall_planes", rclcpp::QoS(1).reliable());
  lidar_wall_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    "/wall_plane_filter/lidar_wall_markers", rclcpp::QoS(1).reliable());
  if (params_.publish_cloud_filtered) {
    const auto cloud_rel = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
    const auto cloud_be = rclcpp::SensorDataQoS().keep_last(5);
    lidar_filtered_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/wall_plane_filter/cloud_filtered", cloud_rel);
    lidar_filtered_pub_be_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/wall_plane_filter/cloud_filtered", cloud_be);
  }
  if (!params_.cloud_topic.empty()) {
    const auto lidar_cb =
      [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {this->cloudCallback(msg);};
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      params_.cloud_topic, rclcpp::SensorDataQoS().keep_last(2).reliable(), lidar_cb);
    cloud_sub_be_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      params_.cloud_topic, rclcpp::SensorDataQoS().keep_last(2).best_effort(), lidar_cb);
  }

  for (const auto & topic : params_.depth_topics) {
    if (topic.empty() || !params_.depth_clean_enable) {continue;}
    std::string cam = topic;
    const auto slash = cam.find_last_of('/');
    if (slash != std::string::npos) {cam = cam.substr(0, slash);}
    if (!cam.empty() && cam.front() == '/') {cam.erase(cam.begin());}
    const auto slash2 = cam.find('/');
    if (slash2 != std::string::npos) {cam = cam.substr(0, slash2);}
    bool find_walls = false;
    for (const auto & name : params_.depth_wall_cameras) {
      if (cam == name) {find_walls = true; break;}
    }
    const auto cloud_rel = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
    const auto cloud_be = rclcpp::SensorDataQoS().keep_last(5);
    const std::string filtered_topic = "/wall_plane_filter/" + cam + "/depth_cloud_filtered";
    const std::string deviate_topic = "/wall_plane_filter/" + cam + "/depth_cloud_deviate";
    depth_filtered_pubs_.push_back(
      create_publisher<sensor_msgs::msg::PointCloud2>(filtered_topic, cloud_rel));
    depth_filtered_pubs_be_.push_back(
      create_publisher<sensor_msgs::msg::PointCloud2>(filtered_topic, cloud_be));
    depth_deviate_pubs_.push_back(
      create_publisher<sensor_msgs::msg::PointCloud2>(deviate_topic, cloud_rel));
    depth_deviate_pubs_be_.push_back(
      create_publisher<sensor_msgs::msg::PointCloud2>(deviate_topic, cloud_be));
    depth_cam_plane_pubs_.push_back(create_publisher<wall_plane_filter::msg::WallPlaneArray>(
      "/wall_plane_filter/" + cam + "/wall_planes", rclcpp::QoS(1).reliable()));
    depth_cam_marker_pubs_.push_back(create_publisher<visualization_msgs::msg::MarkerArray>(
      "/wall_plane_filter/" + cam + "/wall_markers", rclcpp::QoS(1).reliable()));
    depth_cam_names_.push_back(cam);
    depth_find_walls_.push_back(find_walls ? 1 : 0);
    depth_slots_.emplace_back();
    depth_cam_tracks_.emplace_back();
    depth_cam_published_.emplace_back();
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
    "wall_plane_filter: cams=%zu depth_extractor=%s lidar_topic=%s lidar_wall=%d lidar_extractor=%s",
    depth_cam_names_.size(), params_.depth_hough.extractor.c_str(),
    params_.cloud_topic.c_str(), params_.lidar_wall_enable ? 1 : 0,
    params_.lidar_hough.extractor.c_str());
}

void WallPlaneFilterNode::loadParams()
{
  const Params d;
  auto take = [this](const std::string & name, auto & dst, const auto & def) {
    if (!this->has_parameter(name)) {
      this->declare_parameter(name, def);
    }
    this->get_parameter(name, dst);
  };
  take("target_frame", params_.target_frame, d.target_frame);
  take("tf_timeout", params_.tf_timeout, d.tf_timeout);
  take("wall_line_width", params_.wall_line_width, d.wall_line_width);
  take("wall_display_z", params_.wall_display_z, d.wall_display_z);
  take("wall_marker_lifetime", params_.wall_marker_lifetime, d.wall_marker_lifetime);
  take("wall_marker_publish_hz", params_.wall_marker_publish_hz, d.wall_marker_publish_hz);
  take("depth_clean_enable", params_.depth_clean_enable, d.depth_clean_enable);
  take("depth_topics", params_.depth_topics, d.depth_topics);
  take("depth_wall_cameras", params_.depth_wall_cameras, d.depth_wall_cameras);
  take("depth_z_min", params_.depth_z_min, d.depth_z_min);
  take("depth_z_max", params_.depth_z_max, d.depth_z_max);
  take("depth_range_min", params_.depth_range_min, d.depth_range_min);
  take("depth_range_max", params_.depth_range_max, d.depth_range_max);
  take("depth_voxel_xy", params_.depth_voxel_xy, d.depth_voxel_xy);
  take("depth_ror_enable", params_.depth_ror_enable, d.depth_ror_enable);
  take("depth_ror_min_points", params_.depth_ror_min_points, d.depth_ror_min_points);
  take("depth_ror_radius", params_.depth_ror_radius, d.depth_ror_radius);
  take("depth_ror_min_neighbors", params_.depth_ror_min_neighbors, d.depth_ror_min_neighbors);
  take("depth_sor_enable", params_.depth_sor_enable, d.depth_sor_enable);
  take("depth_sor_mean_k", params_.depth_sor_mean_k, d.depth_sor_mean_k);
  take("depth_sor_stddev_mul", params_.depth_sor_stddev_mul, d.depth_sor_stddev_mul);
  take("depth_bilateral_enable", params_.depth_bilateral_enable, d.depth_bilateral_enable);
  take("depth_bilateral_sigma_s", params_.depth_bilateral_sigma_s, d.depth_bilateral_sigma_s);
  take("depth_bilateral_sigma_r", params_.depth_bilateral_sigma_r, d.depth_bilateral_sigma_r);
  take("depth_wall_band_m", params_.depth_wall_band_m, d.depth_wall_band_m);
  take("depth_wall_band_outer_m", params_.depth_wall_band_outer_m, d.depth_wall_band_outer_m);
  take("depth_wall_project", params_.depth_wall_project, d.depth_wall_project);
  take("depth_wall_noise_k", params_.depth_wall_noise_k, d.depth_wall_noise_k);
  take("depth_wall_noise_max", params_.depth_wall_noise_max, d.depth_wall_noise_max);
  take("depth_wall_range_max", params_.depth_wall_range_max, d.depth_wall_range_max);
  take("depth_wall_ground_margin", params_.depth_wall_ground_margin, d.depth_wall_ground_margin);
  take("depth_ground_ref_z", params_.depth_ground_ref_z, d.depth_ground_ref_z);
  take("wall_fuse_min_support_points", params_.wall_fuse_min_support_points,
    d.wall_fuse_min_support_points);
  take("depth_merge_max_age", params_.depth_merge_max_age, d.depth_merge_max_age);
  take("depth_wall_tf_report_sec", params_.depth_wall_tf_report_sec, d.depth_wall_tf_report_sec);
  take("depth_wall_map_frame", params_.depth_wall_map_frame, d.depth_wall_map_frame);
  take("depth_wall_output_frame", params_.depth_wall_output_frame, d.depth_wall_output_frame);
  take("cloud_topic", params_.cloud_topic, d.cloud_topic);
  take("lidar_wall_enable", params_.lidar_wall_enable, d.lidar_wall_enable);
  take("publish_cloud_filtered", params_.publish_cloud_filtered, d.publish_cloud_filtered);
  take("z_min", params_.z_min, d.z_min);
  take("z_max", params_.z_max, d.z_max);
  take("range_min", params_.range_min, d.range_min);
  take("range_max", params_.range_max, d.range_max);
  take("voxel_leaf", params_.voxel_leaf, d.voxel_leaf);
  take("lidar_wall_range_max", params_.lidar_wall_range_max, d.lidar_wall_range_max);
  take("corner_snap_enable", params_.corner_snap_enable, d.corner_snap_enable);
  take("corner_snap_dist", params_.corner_snap_dist, d.corner_snap_dist);
  take("wall_clamp_to_support", params_.wall_clamp_to_support, d.wall_clamp_to_support);
  take("wall_extent_margin_m", params_.wall_extent_margin_m, d.wall_extent_margin_m);

  auto hough = [&](const std::string & prefix, HoughConfig & config) {
    const HoughConfig def;
    take(prefix + ".extractor", config.extractor, def.extractor);
    take(prefix + ".resolution", config.resolution, def.resolution);
    take(prefix + ".half_extent", config.half_extent, def.half_extent);
    take(prefix + ".max_grid_side", config.max_grid_side, def.max_grid_side);
    take(prefix + ".follow_sensor", config.follow_sensor, def.follow_sensor);
    take(prefix + ".origin_x", config.origin_x, def.origin_x);
    take(prefix + ".origin_y", config.origin_y, def.origin_y);
    take(prefix + ".min_cell_points", config.min_cell_points, def.min_cell_points);
    take(prefix + ".min_cell_height", config.min_cell_height, def.min_cell_height);
    take(prefix + ".z_min", config.z_min, def.z_min);
    take(prefix + ".z_max", config.z_max, def.z_max);
    take(prefix + ".theta_deg", config.theta_deg, def.theta_deg);
    take(prefix + ".votes", config.votes, def.votes);
    take(prefix + ".min_line_length", config.min_line_length, def.min_line_length);
    take(prefix + ".hough_max_gap", config.hough_max_gap, def.hough_max_gap);
    take(prefix + ".merge_angle_deg", config.merge_angle_deg, def.merge_angle_deg);
    take(prefix + ".merge_distance", config.merge_distance, def.merge_distance);
    take(prefix + ".merge_gap", config.merge_gap, def.merge_gap);
    take(prefix + ".parallel_angle_deg", config.parallel_angle_deg, def.parallel_angle_deg);
    take(prefix + ".intersection_extension", config.intersection_extension, def.intersection_extension);
    take(prefix + ".support_band", config.support_band, def.support_band);
    take(prefix + ".min_support_ratio", config.min_support_ratio, def.min_support_ratio);
    take(prefix + ".max_unsupported_gap", config.max_unsupported_gap, def.max_unsupported_gap);
    take(prefix + ".min_continuous_support", config.min_continuous_support, def.min_continuous_support);
    take(prefix + ".min_support_points", config.min_support_points, def.min_support_points);
    take(prefix + ".max_lines", config.max_lines, def.max_lines);
    take(prefix + ".refine", config.refine, def.refine);
    take(prefix + ".debug", config.debug, def.debug);
    take(prefix + ".ransac_distance", config.ransac_distance, def.ransac_distance);
    take(prefix + ".ransac_min_inliers", config.ransac_min_inliers, def.ransac_min_inliers);
    take(prefix + ".ransac_max_iterations", config.ransac_max_iterations, def.ransac_max_iterations);
    take(prefix + ".ransac_downsample_leaf", config.ransac_downsample_leaf, def.ransac_downsample_leaf);
    take(prefix + ".seed_contour_angle_deg", config.seed_contour_angle_deg, def.seed_contour_angle_deg);
    take(prefix + ".seed_min_points", config.seed_min_points, def.seed_min_points);
    take(prefix + ".seed_max_distance", config.seed_max_distance, def.seed_max_distance);
    take(prefix + ".grow_distance", config.grow_distance, def.grow_distance);
    take(prefix + ".grow_max_gap_bins", config.grow_max_gap_bins, def.grow_max_gap_bins);
    take(prefix + ".grow_max_range_jump", config.grow_max_range_jump, def.grow_max_range_jump);
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
    take("wall_track.coast_gate_extra", t.coast_gate_extra, td.coast_gate_extra);
    take("wall_track.clamp_extent_to_obs", t.clamp_extent_to_obs, td.clamp_extent_to_obs);
    take("wall_track.kf_enable", t.kf_enable, td.kf_enable);
    take("wall_track.kf_q_rho", t.kf_q_rho, td.kf_q_rho);
    take("wall_track.kf_q_theta", t.kf_q_theta, td.kf_q_theta);
    take("wall_track.kf_q_end", t.kf_q_end, td.kf_q_end);
    take("wall_track.kf_r_rho", t.kf_r_rho, td.kf_r_rho);
    take("wall_track.kf_r_theta", t.kf_r_theta, td.kf_r_theta);
    take("wall_track.kf_r_end", t.kf_r_end, td.kf_r_end);
    if (!(t.lose_sec > 0.0) || t.confirm_frames < 1 || t.shorten_frames < 1) {
      throw std::invalid_argument("Invalid wall_track parameters");
    }
  }

  if (!(params_.depth_range_max > 0.0) || !(params_.depth_z_max > params_.depth_z_min) ||
    !(params_.depth_voxel_xy > 0.0) || params_.depth_sor_mean_k < 3 ||
    !(params_.depth_ror_radius > 0.0) || params_.depth_ror_min_neighbors < 1 ||
    params_.depth_ror_min_points < 1 ||
    !(params_.depth_bilateral_sigma_s > 0.0) || !(params_.depth_bilateral_sigma_r > 0.0) ||
    !(params_.wall_marker_lifetime > 0.0) || !(params_.z_max > params_.z_min) ||
    !(params_.range_max > params_.range_min) || !(params_.voxel_leaf > 0.0) ||
    !(params_.lidar_wall_range_max > 0.0) || !(params_.wall_extent_margin_m >= 0.0) ||
    !(params_.corner_snap_dist >= 0.0))
  {
    throw std::invalid_argument("Invalid depth/lidar preprocess / marker parameters");
  }
  validateHoughConfig(params_.depth_hough);
  validateHoughConfig(params_.lidar_hough);
}

std::string WallPlaneFilterNode::wallWorkFrame() const
{
  return params_.depth_wall_map_frame.empty() ? "map" : params_.depth_wall_map_frame;
}

std::string WallPlaneFilterNode::outputFrame() const
{
  return params_.depth_wall_output_frame.empty() ?
    wallWorkFrame() : params_.depth_wall_output_frame;
}

bool WallPlaneFilterNode::lookupIso(
  const std::string & to_frame, const std::string & from_frame, const rclcpp::Time & stamp,
  Eigen::Isometry3d & out, std::string * reason)
{
  if (reason) {reason->clear();}
  if (to_frame.empty() || from_frame.empty()) {
    if (reason) {
      *reason = "empty frame (to='" + to_frame + "', from='" + from_frame + "')";
    }
    return false;
  }
  if (to_frame == from_frame) {
    out = Eigen::Isometry3d::Identity();
    return true;
  }
  try {
    const auto tf = tf_buffer_->lookupTransform(
      to_frame, from_frame, stamp, rclcpp::Duration::from_seconds(params_.tf_timeout));
    out = tf2::transformToEigen(tf.transform);
    return true;
  } catch (const std::exception & e) {
    if (reason) {*reason = e.what();}
    return false;
  }
}

bool WallPlaneFilterNode::transformCloud(
  const CloudT & in, const std::string & to_frame, const std::string & from_frame,
  const rclcpp::Time & stamp, CloudT & out, std::string * reason)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  if (!(to_frame == from_frame || lookupIso(to_frame, from_frame, stamp, T, reason))) {
    if (reason && reason->empty()) {
      *reason = "lookup failed " + from_frame + " -> " + to_frame;
    }
    return false;
  }
  if (reason) {reason->clear();}
  transformInto(in, T, out);
  return true;
}

bool WallPlaneFilterNode::acceptStampOnce(int key, int64_t stamp_ns)
{
  std::lock_guard<std::mutex> lock(dedup_mutex_);
  for (const auto & p : recent_stamps_) {
    if (p.first == key && p.second == stamp_ns) {return false;}
  }
  recent_stamps_.emplace_back(key, stamp_ns);
  while (recent_stamps_.size() > kDedupHistory) {recent_stamps_.pop_front();}
  return true;
}

sensor_msgs::msg::PointCloud2 WallPlaneFilterNode::toMsg(
  const CloudT & cloud, const rclcpp::Time & stamp, const std::string & frame) const
{
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  msg.header.stamp = stamp;
  msg.header.frame_id = frame;
  return msg;
}

void WallPlaneFilterNode::publishCloud(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & reliable,
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & best_effort,
  const sensor_msgs::msg::PointCloud2 & msg)
{
  if (reliable) {reliable->publish(msg);}
  if (best_effort) {best_effort->publish(msg);}
}

void WallPlaneFilterNode::clearWallCachesOnTimeJump(const rclcpp::Time & stamp)
{
  const double tol = 0.05;
  if (wall_path_stamp_.nanoseconds() == 0) {
    wall_path_stamp_ = stamp; return;
  }
  const double dt = (stamp - wall_path_stamp_).seconds();
  if (dt >= 0.0) {wall_path_stamp_ = stamp; return;}
  if (-dt <= tol) {return;}
  {
    std::lock_guard<std::mutex> lock(depth_wall_mutex_);
    for (auto & slot : depth_slots_) {slot = DepthSlot();}
    for (auto & t : depth_cam_tracks_) {t.clear();}
    for (auto & lines : depth_cam_published_) {lines.clear();}
    validated_depth_lines_.clear();
    validated_depth_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  }
  {
    std::lock_guard<std::mutex> lock(lidar_wall_mutex_);
    lidar_tracks_.clear();
  }
  wall_path_stamp_ = stamp;
}

void WallPlaneFilterNode::reportDepthTf()
{
  const double period = std::max(1.0, params_.depth_wall_tf_report_sec);
  const auto now = std::chrono::steady_clock::now();
  if (std::chrono::duration<double>(now - depth_tf_report_last_).count() < period) {return;}
  depth_tf_report_last_ = now;
  std::lock_guard<std::mutex> lock(depth_tf_mutex_);
  for (size_t i = 0; i < depth_cam_names_.size(); ++i) {
    if (depth_tf_total_[i] == 0) {continue;}
    RCLCPP_INFO(
      get_logger(), "depth tf camera=%s fail=%zu/%zu",
      depth_cam_names_[i].c_str(), depth_tf_fail_[i], depth_tf_total_[i]);
  }
}

void WallPlaneFilterNode::cloudCallback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  if (!msg || params_.cloud_topic.empty()) {return;}
  const rclcpp::Time stamp(msg->header.stamp);
  clearWallCachesOnTimeJump(stamp);
  // key -1: lidar path (depth cams use 0..N-1)
  if (!acceptStampOnce(-1, stamp.nanoseconds())) {return;}

  const std::string world = wallWorkFrame();
  const rclcpp::Time lookup_stamp = stamp.nanoseconds() == 0 ?
    rclcpp::Time(0, 0, get_clock()->get_clock_type()) : stamp;

  Eigen::Isometry3d T_bc = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_wb = Eigen::Isometry3d::Identity();
  std::string tf_reason;
  if (!lookupIso(params_.target_frame, msg->header.frame_id, lookup_stamp, T_bc, &tf_reason)) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "lidar TF fail stage=sensor->%s from=%s stamp=%.3f reason=%s (skip frame, no cloud publish)",
      params_.target_frame.c_str(), msg->header.frame_id.c_str(), stamp.seconds(),
      tf_reason.c_str());
    return;
  }
  if (!lookupIso(world, params_.target_frame, lookup_stamp, T_wb, &tf_reason)) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "lidar TF fail stage=%s->%s stamp=%.3f reason=%s (skip frame, no cloud publish)",
      params_.target_frame.c_str(), world.c_str(), stamp.seconds(), tf_reason.c_str());
    return;
  }

  CloudT raw;
  pcl::fromROSMsg(*msg, raw);

  // Simple ROI in target_frame (base_link).
  CloudT roi;
  roi.reserve(raw.size());
  for (const auto & p : raw.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {continue;}
    const Eigen::Vector3d q = T_bc * Eigen::Vector3d(p.x, p.y, p.z);
    if (q.z() < params_.z_min || q.z() > params_.z_max) {continue;}
    const double r = std::hypot(q.x(), q.y());
    if (r < params_.range_min || r > params_.range_max) {continue;}
    roi.push_back(pcl::PointXYZ(
      static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z())));
  }
  finishCloud(roi);

  CloudT filtered = roi;
  if (!roi.empty() && params_.voxel_leaf > 1e-9) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(roi.makeShared());
    vg.setLeafSize(
      static_cast<float>(params_.voxel_leaf),
      static_cast<float>(params_.voxel_leaf),
      static_cast<float>(params_.voxel_leaf));
    vg.filter(filtered);
    finishCloud(filtered);
  }

  if (params_.publish_cloud_filtered && (lidar_filtered_pub_ || lidar_filtered_pub_be_)) {
    CloudT filtered_world;
    transformInto(filtered, T_wb, filtered_world);
    const std::string out_frame = outputFrame();
    CloudT out;
    std::string out_reason;
    if (transformCloud(filtered_world, out_frame, world, stamp, out, &out_reason)) {
      publishCloud(lidar_filtered_pub_, lidar_filtered_pub_be_, toMsg(out, stamp, out_frame));
    } else {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "lidar filtered cloud TF fail from=%s to=%s stamp=%.3f reason=%s (skip publish)",
        world.c_str(), out_frame.c_str(), stamp.seconds(), out_reason.c_str());
    }
  }

  if (!params_.lidar_wall_enable) {return;}

  CloudT wall_base;
  wall_base.reserve(filtered.size());
  for (const auto & p : filtered.points) {
    if (p.z < params_.lidar_hough.z_min || p.z > params_.lidar_hough.z_max) {continue;}
    if (std::hypot(p.x, p.y) > params_.lidar_wall_range_max) {continue;}
    wall_base.push_back(p);
  }
  finishCloud(wall_base);
  CloudT wall_world;
  transformInto(wall_base, T_wb, wall_world);
  const Eigen::Vector2d robot_xy = T_wb.translation().head<2>();

  const auto result = extractHoughWalls(wall_world, robot_xy, params_.lidar_hough);
  std::vector<WallLine> observations = result.lines;
  finalizeWallGeometry(observations, &wall_world, params_.lidar_hough, params_);
  const WallTrackConfig track_cfg = trackConfigFrom(params_.lidar_hough, params_.wall_track);
  std::vector<WallLine> tracked;
  WallTrackUpdateStats stats;
  {
    std::lock_guard<std::mutex> lock(lidar_wall_mutex_);
    tracked = lidar_tracks_.update(observations, stamp.seconds(), track_cfg, &stats);
  }
  finalizeWallGeometry(tracked, &wall_world, params_.lidar_hough, params_);
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 2000,
    "lidar wall track obs=%d matched=%d coast=%d expire=%d out=%d extract=%zu extractor=%s",
    stats.obs, stats.matched, stats.coasting, stats.expired, stats.confirmed_out,
    result.lines.size(), params_.lidar_hough.extractor.c_str());
  publishLidarWalls(tracked, stamp, world);
}

void WallPlaneFilterNode::publishLidarWalls(
  const std::vector<WallLine> & lines, const rclcpp::Time & stamp, const std::string & frame)
{
  if (!lidar_wall_plane_pub_ || !lidar_wall_marker_pub_) {return;}
  const std::string output = outputFrame();
  wall_plane_filter::msg::WallPlaneArray planes;
  planes.header.frame_id = output;
  planes.header.stamp = stamp;
  visualization_msgs::msg::MarkerArray markers;
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  const bool tf_ok = output == frame || lookupIso(output, frame, stamp, T);
  using Marker = visualization_msgs::msg::Marker;
  Marker edges;
  edges.header = planes.header;
  edges.ns = "lidar_walls";
  edges.id = 0;
  edges.type = Marker::LINE_LIST;
  edges.action = Marker::ADD;
  edges.pose.orientation.w = 1.0;
  edges.scale.x = params_.wall_line_width;
  edges.color.r = 0.15f; edges.color.g = 0.95f; edges.color.b = 0.35f; edges.color.a = 1.0f;
  edges.lifetime = rclcpp::Duration::from_seconds(params_.wall_marker_lifetime);
  Marker faces = edges;
  faces.ns = "lidar_wall_planes";
  faces.id = 1;
  faces.type = Marker::TRIANGLE_LIST;
  faces.scale.x = faces.scale.y = faces.scale.z = 1.0;
  faces.color.a = 0.30f;

  if (tf_ok) {
    for (const auto & line_in : lines) {
      WallLine line = (output == frame) ? line_in : transformLine(line_in, T);
      planes.planes.push_back(
        toWallPlaneMsg(line, params_.lidar_hough.z_min, params_.lidar_hough.z_max));
      const auto & pl = planes.planes.back();
      edges.points.push_back(pl.corners[0]); edges.points.push_back(pl.corners[1]);
      edges.points.push_back(pl.corners[3]); edges.points.push_back(pl.corners[2]);
      faces.points.push_back(pl.corners[0]); faces.points.push_back(pl.corners[1]);
      faces.points.push_back(pl.corners[2]);
      faces.points.push_back(pl.corners[0]); faces.points.push_back(pl.corners[2]);
      faces.points.push_back(pl.corners[3]);
    }
  }
  lidar_wall_plane_pub_->publish(planes);
  // Always publish markers (empty ADD clears stale when no walls).
  if (edges.points.empty()) {
    Marker del;
    del.header = planes.header;
    del.ns = "lidar_walls";
    del.id = 0;
    del.action = Marker::DELETE;
    markers.markers.push_back(del);
    del.ns = "lidar_wall_planes";
    del.id = 1;
    markers.markers.push_back(del);
  } else {
    markers.markers.push_back(edges);
    markers.markers.push_back(faces);
  }
  lidar_wall_marker_pub_->publish(markers);
}

void WallPlaneFilterNode::depthCallback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg, size_t idx)
{
  if (!msg || idx >= depth_slots_.size()) {return;}
  const rclcpp::Time stamp(msg->header.stamp);
  clearWallCachesOnTimeJump(stamp);
  if (!acceptStampOnce(static_cast<int>(idx), stamp.nanoseconds())) {return;}

  const bool find_walls = depth_find_walls_[idx] != 0;
  const std::string world = wallWorkFrame();
  const rclcpp::Time lookup_stamp = stamp.nanoseconds() == 0 ?
    rclcpp::Time(0, 0, get_clock()->get_clock_type()) : stamp;

  {
    std::lock_guard<std::mutex> lock(depth_tf_mutex_);
    ++depth_tf_total_[idx];
  }
  Eigen::Isometry3d T_bc = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_wb = Eigen::Isometry3d::Identity();
  std::string base_reason;
  std::string map_reason;
  const bool base_ok = lookupIso(
    params_.target_frame, msg->header.frame_id, lookup_stamp, T_bc, &base_reason);
  const bool map_ok = base_ok &&
    lookupIso(world, params_.target_frame, lookup_stamp, T_wb, &map_reason);
  if (!map_ok) {
    std::lock_guard<std::mutex> lock(depth_tf_mutex_);
    ++depth_tf_fail_[idx];
  }
  reportDepthTf();
  if (!base_ok) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "depth TF fail camera=%s stage=sensor->%s from=%s to=%s world=%s stamp=%.3f "
      "reason=%s (skip frame, no cloud publish)",
      depth_cam_names_[idx].c_str(), params_.target_frame.c_str(), msg->header.frame_id.c_str(),
      params_.target_frame.c_str(), world.c_str(), stamp.seconds(), base_reason.c_str());
    return;
  }
  if (!map_ok) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "depth TF fail camera=%s stage=%s->%s from=%s to=%s world=%s stamp=%.3f "
      "reason=%s (skip map wall + no cloud publish)",
      depth_cam_names_[idx].c_str(), params_.target_frame.c_str(), world.c_str(),
      msg->header.frame_id.c_str(), params_.target_frame.c_str(), world.c_str(),
      stamp.seconds(), map_reason.c_str());
    return;
  }

  CloudT raw;
  pcl::fromROSMsg(*msg, raw);
  CloudT filtered, deviate;
  PrepStats prep{};
  preprocessDepthCloud(raw, T_bc, params_, filtered, deviate, prep);
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 2000,
    "depth preprocess camera=%s crop=%zu voxel_xy=%zu ror_drop=%zu sor_drop=%zu bilateral=%d "
    "out=%zu z=[%.2f,%.2f] range=%.2f xy=%.3f",
    depth_cam_names_[idx].c_str(), prep.crop, prep.voxel, prep.ror_drop, prep.sor_drop,
    prep.bilateral ? 1 : 0, prep.out, params_.depth_z_min, params_.depth_z_max,
    params_.depth_range_max, params_.depth_voxel_xy);

  DepthSlot slot;
  transformInto(filtered, T_wb, slot.filtered_world);
  transformInto(deviate, T_wb, slot.deviate_world);
  slot.robot_xy = T_wb.translation().head<2>();
  slot.sensor_xy = (T_wb * T_bc).translation().head<2>();
  slot.stamp = stamp;

  if (find_walls) {
    const double floor_cut = params_.depth_ground_ref_z + params_.depth_wall_ground_margin;
    const Eigen::Vector3d cam_b = T_bc.translation();
    const double noise_k = std::max(0.0, params_.depth_wall_noise_k);
    CloudT wall_base;
    wall_base.reserve(filtered.size());
    for (const auto & p : filtered.points) {
      if (p.z < std::max(floor_cut, params_.depth_hough.z_min) ||
        p.z > params_.depth_hough.z_max || std::hypot(p.x, p.y) > params_.depth_wall_range_max)
      {
        continue;
      }
      const double d = (Eigen::Vector3d(p.x, p.y, p.z) - cam_b).norm();
      if (noise_k * d * d > params_.depth_wall_noise_max) {continue;}
      wall_base.push_back(p);
    }
    finishCloud(wall_base);
    CloudT wall_world;
    transformInto(wall_base, T_wb, wall_world);
    processDepthWalls(idx, wall_world, slot.robot_xy, stamp);
  }

  publishDepthCloud(idx, slot);
  std::lock_guard<std::mutex> lock(depth_wall_mutex_);
  depth_slots_[idx] = std::move(slot);
}

void WallPlaneFilterNode::processDepthWalls(
  size_t idx, const CloudT & wall_world, const Eigen::Vector2d & robot_xy,
  const rclcpp::Time & stamp)
{
  if (idx >= depth_cam_tracks_.size()) {return;}
  const auto result = extractHoughWalls(wall_world, robot_xy, params_.depth_hough);
  std::vector<WallLine> observations = result.lines;
  finalizeWallGeometry(observations, &wall_world, params_.depth_hough, params_);

  {
    std::vector<ResidualStats> per_wall;
    ResidualStats global;
    std::vector<double> all_abs;
    collectWallSupportResiduals(
      observations, wall_world, params_.depth_hough.support_band, per_wall, global, all_abs);
    double p95_abs = 0.0;
    if (!all_abs.empty()) {
      std::vector<double> sorted = all_abs;
      const size_t k = static_cast<size_t>(
        std::min(sorted.size() - 1, static_cast<size_t>(0.95 * (sorted.size() - 1))));
      std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(k), sorted.end());
      p95_abs = sorted[k];
    }
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "wall residual %s walls=%zu pts=%zu |e| min=%.4f max=%.4f mean=%.4f rms=%.4f "
      "std=%.4f p95=%.4f signed min=%.4f max=%.4f mean=%.4f",
      depth_cam_names_[idx].c_str(), observations.size(), global.count,
      global.min_abs, global.max_abs, global.mean_abs, global.rms, global.std_abs, p95_abs,
      global.min_e, global.max_e, global.mean_e);
    for (size_t wi = 0; wi < observations.size() && wi < per_wall.size(); ++wi) {
      const auto & w = observations[wi];
      const auto & s = per_wall[wi];
      if (s.count == 0) {continue;}
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "wall residual %s[%zu] len=%.2f support=%zu/%zu |e| min=%.4f max=%.4f mean=%.4f "
        "rms=%.4f sigma=%.4f signed min=%.4f max=%.4f mean=%.4f",
        depth_cam_names_[idx].c_str(), wi, (w.b - w.a).norm(), s.count, w.support_points,
        s.min_abs, s.max_abs, s.mean_abs, s.rms, w.sigma, s.min_e, s.max_e, s.mean_e);
    }
  }

  const WallTrackConfig track_cfg = trackConfigFrom(params_.depth_hough, params_.wall_track);
  std::vector<WallLine> cam_tracked;
  WallTrackUpdateStats stats;
  {
    std::lock_guard<std::mutex> lock(depth_wall_mutex_);
    cam_tracked = depth_cam_tracks_[idx].update(
      observations, stamp.seconds(), track_cfg, &stats);
    finalizeWallGeometry(cam_tracked, &wall_world, params_.depth_hough, params_);
    if (idx < depth_cam_published_.size()) {
      depth_cam_published_[idx] = cam_tracked;
    }
    std::vector<WallLine> all;
    for (const auto & lines : depth_cam_published_) {
      all.insert(all.end(), lines.begin(), lines.end());
    }
    validated_depth_lines_ = fuseWallLines(
      all, params_.depth_hough.merge_angle_deg, params_.depth_hough.merge_distance,
      params_.depth_hough.merge_gap, params_.wall_fuse_min_support_points);
    // Fused multi-cam set: corner-lock only (no single support cloud).
    if (params_.corner_snap_enable) {
      const double snap = params_.corner_snap_dist > 1e-9 ?
        params_.corner_snap_dist : params_.depth_hough.intersection_extension;
      snapWallCorners(
        validated_depth_lines_, params_.depth_hough.parallel_angle_deg, snap,
        params_.depth_hough.min_line_length);
    }
    validated_depth_stamp_ = stamp;
  }
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 2000,
    "wall track %s obs=%d matched=%d coast=%d expire=%d out=%d extract=%zu",
    depth_cam_names_[idx].c_str(), stats.obs, stats.matched, stats.coasting, stats.expired,
    stats.confirmed_out, result.lines.size());
  publishCamWalls(idx, cam_tracked, stamp, wallWorkFrame());
}

void WallPlaneFilterNode::publishCamWalls(
  size_t idx, const std::vector<WallLine> & lines, const rclcpp::Time & stamp,
  const std::string & frame)
{
  if (idx >= depth_cam_plane_pubs_.size()) {return;}
  const std::string output = outputFrame();
  wall_plane_filter::msg::WallPlaneArray planes;
  planes.header.frame_id = output;
  planes.header.stamp = stamp;
  visualization_msgs::msg::MarkerArray markers;
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  const bool tf_ok = output == frame || lookupIso(output, frame, stamp, T);
  static const float kRgb[][3] = {
    {0.20f, 1.00f, 0.90f}, {1.00f, 0.85f, 0.20f},
    {1.00f, 0.40f, 0.90f}, {0.35f, 0.65f, 1.00f}};
  const auto & rgb = kRgb[idx % 4];
  const std::string ns = depth_cam_names_[idx] + "_walls";
  using Marker = visualization_msgs::msg::Marker;
  Marker edges;
  edges.header = planes.header;
  edges.ns = ns;
  edges.id = 0;
  edges.type = Marker::LINE_LIST;
  edges.action = Marker::ADD;
  edges.pose.orientation.w = 1.0;
  edges.scale.x = params_.wall_line_width;
  edges.color.r = rgb[0]; edges.color.g = rgb[1]; edges.color.b = rgb[2]; edges.color.a = 1.0f;
  edges.lifetime = rclcpp::Duration::from_seconds(params_.wall_marker_lifetime);
  Marker faces = edges;
  faces.ns = depth_cam_names_[idx] + "_wall_planes";
  faces.id = 1;
  faces.type = Marker::TRIANGLE_LIST;
  faces.scale.x = faces.scale.y = faces.scale.z = 1.0;
  faces.color.a = 0.35f;

  if (tf_ok) {
    for (const auto & line_in : lines) {
      WallLine line = (output == frame) ? line_in : transformLine(line_in, T);
      planes.planes.push_back(
        toWallPlaneMsg(line, params_.depth_hough.z_min, params_.depth_hough.z_max));
      const auto & pl = planes.planes.back();
      edges.points.push_back(pl.corners[0]); edges.points.push_back(pl.corners[1]);
      edges.points.push_back(pl.corners[3]); edges.points.push_back(pl.corners[2]);
      faces.points.push_back(pl.corners[0]); faces.points.push_back(pl.corners[1]);
      faces.points.push_back(pl.corners[2]);
      faces.points.push_back(pl.corners[0]); faces.points.push_back(pl.corners[2]);
      faces.points.push_back(pl.corners[3]);
    }
  }
  depth_cam_plane_pubs_[idx]->publish(planes);
  if (!edges.points.empty()) {
    markers.markers.push_back(edges);
    markers.markers.push_back(faces);
    depth_cam_marker_pubs_[idx]->publish(markers);
  }
}

void WallPlaneFilterNode::publishDepthCloud(size_t idx, const DepthSlot & slot)
{
  std::vector<WallLine> walls;
  {
    std::lock_guard<std::mutex> lock(depth_wall_mutex_);
    if (idx < depth_cam_published_.size() && !depth_cam_published_[idx].empty()) {
      walls = depth_cam_published_[idx];
    } else if (idx < depth_cam_tracks_.size()) {
      walls = depth_cam_tracks_[idx].confirmed();
    }
    const bool stamp_ok = validated_depth_stamp_.nanoseconds() != 0 &&
      std::abs((slot.stamp - validated_depth_stamp_).seconds()) <=
      std::max(params_.depth_merge_max_age, 0.5);
    if (stamp_ok && !validated_depth_lines_.empty()) {
      walls = validated_depth_lines_;
    }
  }
  CloudT filtered = slot.filtered_world;
  CloudT deviate = slot.deviate_world;
  size_t proj = 0;
  if (!walls.empty()) {
    cullWallBand(
      filtered, deviate, walls, params_.depth_wall_band_m, params_.depth_wall_band_outer_m,
      params_.depth_wall_project, slot.robot_xy, params_.depth_wall_noise_k,
      params_.depth_wall_noise_max, &slot.sensor_xy, &proj);
  }
  const std::string world = wallWorkFrame();
  const std::string out_frame = outputFrame();
  CloudT filtered_out, deviate_out;
  std::string filt_reason;
  std::string dev_reason;
  const bool filt_ok = transformCloud(
    filtered, out_frame, world, slot.stamp, filtered_out, &filt_reason);
  const bool dev_ok = transformCloud(
    deviate, out_frame, world, slot.stamp, deviate_out, &dev_reason);
  if (!(filt_ok && dev_ok)) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "depth cloud publish TF fail camera=%s idx=%zu from=%s to=%s stamp=%.3f "
      "filtered_ok=%d reason=%s deviate_ok=%d reason=%s (skip publish)",
      idx < depth_cam_names_.size() ? depth_cam_names_[idx].c_str() : "?",
      idx, world.c_str(), out_frame.c_str(), slot.stamp.seconds(),
      filt_ok ? 1 : 0, filt_reason.c_str(), dev_ok ? 1 : 0, dev_reason.c_str());
    return;
  }
  publishCloud(
    depth_filtered_pubs_[idx], depth_filtered_pubs_be_[idx],
    toMsg(filtered_out, slot.stamp, out_frame));
  publishCloud(
    depth_deviate_pubs_[idx], depth_deviate_pubs_be_[idx],
    toMsg(deviate_out, slot.stamp, out_frame));
}

}  // namespace wall_plane_filter
