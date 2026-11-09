#include "wall_plane_filter/wall_plane_filter_node.hpp"
#include "wall_plane_filter/lidar_tail_filter.hpp"
#include "wall_plane_filter/wall_hough.hpp"
#include "wall_plane_filter/wall_track.hpp"

#include <tf2_eigen/tf2_eigen.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
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

// Flying pixels, 3x3 range median, then stride decimation. Neighborhood is the
// depth image, so this only runs on an organized cloud. Output points stay on
// the original camera ray. fly receives the raw pixels rejected as edges.
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
    const int du[4] = {1, -1, 0, 0};
    const int dv[4] = {0, 0, 1, -1};
    for (int v = 0; v < h; ++v) {
      for (int u = 0; u < w; ++u) {
        const size_t i = static_cast<size_t>(v * w + u);
        const float r = range[i];
        if (!std::isfinite(r)) {
          continue;
        }
        const float lim = std::max(abs_m, rel * r);
        for (int k = 0; k < 4; ++k) {
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
            flying[i] = 1;
            break;
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
    const int need = std::max(1, params.depth_median_min);
    float window[9];
    for (int v = 0; v < h; ++v) {
      for (int u = 0; u < w; ++u) {
        const size_t i = static_cast<size_t>(v * w + u);
        if (flying[i] || !std::isfinite(range[i])) {
          med[i] = std::numeric_limits<float>::quiet_NaN();
          continue;
        }
        int count = 0;
        for (int dv = -1; dv <= 1; ++dv) {
          for (int du = -1; du <= 1; ++du) {
            const int uu = u + du;
            const int vv = v + dv;
            if (uu < 0 || vv < 0 || uu >= w || vv >= h) {
              continue;
            }
            const size_t j = static_cast<size_t>(vv * w + uu);
            if (flying[j] || !std::isfinite(range[j])) {
              continue;
            }
            window[count++] = range[j];
          }
        }
        if (count < need) {
          med[i] = std::numeric_limits<float>::quiet_NaN();
          continue;
        }
        std::nth_element(window, window + count / 2, window + count);
        med[i] = window[count / 2];
      }
    }
  }

  const int stride = std::max(1, params.depth_decimate_stride);
  kept.reserve(n / static_cast<size_t>(stride * stride));
  for (int v = 0; v < h; v += stride) {
    for (int u = 0; u < w; u += stride) {
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

double segDist2(const Eigen::Vector2d & p, const Eigen::Vector2d & a, const Eigen::Vector2d & b)
{
  const Eigen::Vector2d ab = b - a;
  const double len2 = ab.squaredNorm();
  double t = 0.0;
  if (len2 > 1e-12) {
    t = std::clamp((p - a).dot(ab) / len2, 0.0, 1.0);
  }
  return (p - (a + t * ab)).norm();
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

double distToPoly(const Eigen::Vector2d & p, const std::vector<Eigen::Vector2d> & xy)
{
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = 1; i < xy.size(); ++i) {
    best = std::min(best, segDist2(p, xy[i - 1], xy[i]));
  }
  return best;
}

// Points in (keep, outer] of a published wall are the smear around the surface.
// Points farther than outer are obstacles and are kept.
size_t cullWallBand(
  CloudT & cloud, CloudT & deviate, const std::vector<std::vector<Eigen::Vector2d>> & walls,
  double keep, double outer)
{
  if (cloud.empty() || walls.empty() || !(keep > 0.0) || !(outer > keep)) {
    return 0;
  }
  CloudT kept;
  kept.reserve(cloud.size());
  size_t dropped = 0;
  for (const auto & p : cloud.points) {
    const Eigen::Vector2d xy(p.x, p.y);
    double best = std::numeric_limits<double>::infinity();
    for (const auto & wall : walls) {
      best = std::min(best, distToPoly(xy, wall));
    }
    if (best > keep && best <= outer) {
      deviate.push_back(p);
      ++dropped;
    } else {
      kept.push_back(p);
    }
  }
  cloud = std::move(kept);
  finishCloud(cloud);
  finishCloud(deviate);
  return dropped;
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
    depth_cam_names_.push_back(cam);
    depth_find_walls_.push_back(find_walls ? 1 : 0);
    depth_slots_.emplace_back();
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

  // Node clock follows use_sim_time, so a paused bag pauses fusion too.
  if (params_.depth_wall_fuse_hz > 1e-3) {
    wall_fuse_timer_ = rclcpp::create_timer(
      this, get_clock(),
      rclcpp::Duration::from_seconds(1.0 / params_.depth_wall_fuse_hz),
      [this]() {wallFuseTick();});
  }

  RCLCPP_INFO(
    get_logger(),
    "lidar z-cut [%.2f, %.2f] range<%.2f wall_edge=%d cloud=%s",
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
  take("depth_fly_enable", params_.depth_fly_enable, d.depth_fly_enable);
  take("depth_fly_abs_m", params_.depth_fly_abs_m, d.depth_fly_abs_m);
  take("depth_fly_rel", params_.depth_fly_rel, d.depth_fly_rel);
  take("depth_median_enable", params_.depth_median_enable, d.depth_median_enable);
  take("depth_median_min", params_.depth_median_min, d.depth_median_min);
  take("depth_decimate_stride", params_.depth_decimate_stride, d.depth_decimate_stride);
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
  wall_plane_filter::msg::WallPlaneArray empty_planes;
  empty_planes.header = marker.header;
  depth_wall_plane_pub_->publish(empty_planes);
  lidar_wall_plane_pub_->publish(empty_planes);
  validated_depth_lines_.clear();
  validated_depth_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  depth_tracks_.clear();
  lidar_tracks_.clear();
  depth_grid_valid_ = false;
  wall_data_dirty_ = false;
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

void WallPlaneFilterNode::cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  if (!msg) {
    return;
  }
  const rclcpp::Time stamp(msg->header.stamp);
  clearWallCachesOnTimeJump(stamp);
  if (params_.process_hz > 1e-3 && last_stamp_.nanoseconds() > 0) {
    const double dt = (stamp - last_stamp_).seconds();
    if (dt >= 0.0 && dt < 1.0 / params_.process_hz) {
      return;
    }
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
  last_stamp_ = cloud_stamp;
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
      tracked = lidar_tracks_.update(map_obs, cloud_stamp.seconds(), trackConfigFrom(params_.lidar_hough));
    }
    publishTrackedMarkers(tracked, map_corners, cloud_stamp, map_frame, false);
  }

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
  const bool find_walls = depth_find_walls_[idx] != 0;
  const std::string world = wallWorkFrame();
  // A zero stamp cannot be looked up. Use the latest TF only in that case.
  const rclcpp::Time lookup_stamp = stamp.nanoseconds() == 0 ?
    rclcpp::Time(0, 0, get_clock()->get_clock_type()) : stamp;

  ++depth_tf_total_[idx];
  Eigen::Isometry3d T_bc = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_wb = Eigen::Isometry3d::Identity();
  const bool base_ok = lookupIso(params_.target_frame, msg->header.frame_id, lookup_stamp, T_bc);
  const bool map_ok = base_ok && lookupIso(world, params_.target_frame, lookup_stamp, T_wb);
  if (!map_ok) {
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
  if (!find_walls) {
    DepthSlot slot;
    transformInto(filtered, T_wb, slot.filtered_world);
    transformInto(deviate, T_wb, slot.deviate_world);
    slot.robot_xy = T_wb.translation().head<2>();
    slot.stamp = stamp;
    slot.ready_steady = std::chrono::steady_clock::now();
    slot.fresh = true;
    publishDepthCloud(idx, slot);
    std::lock_guard<std::mutex> lock(depth_wall_mutex_);
    depth_slots_[idx] = std::move(slot);
    wall_data_dirty_ = true;
    return;
  }

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
      p.z > params_.depth_hough.z_max || std::hypot(p.x, p.y) > params_.depth_wall_range_max) {
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

  DepthSlot slot;
  transformInto(wall_base, T_wb, slot.wall_world);
  transformInto(filtered, T_wb, slot.filtered_world);
  transformInto(deviate, T_wb, slot.deviate_world);
  slot.origin_world = T_wb * cam_b;
  slot.robot_xy = T_wb.translation().head<2>();
  slot.stamp = stamp;
  slot.ready_steady = std::chrono::steady_clock::now();
  slot.fresh = true;
  publishDepthCloud(idx, slot);
  std::lock_guard<std::mutex> lock(depth_wall_mutex_);
  depth_slots_[idx] = std::move(slot);
  wall_data_dirty_ = true;
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
  std::vector<std::vector<Eigen::Vector2d>> wall_polys;
  {
    std::lock_guard<std::mutex> lock(depth_wall_mutex_);
    // validated_depth_lines_ are the smoothed map tracks, including this fusion tick.
    if (validated_depth_stamp_.nanoseconds() != 0 &&
      std::abs((slot.stamp - validated_depth_stamp_).seconds()) <= params_.depth_merge_max_age)
    {
      for (const WallLine & line : validated_depth_lines_) {
        wall_polys.push_back({line.a, line.b});
      }
    }
  }
  CloudT filtered = slot.filtered_world;
  CloudT deviate = slot.deviate_world;
  cullWallBand(
    filtered, deviate, wall_polys,
    params_.depth_wall_band_m, params_.depth_wall_band_outer_m);
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
  const auto now = std::chrono::steady_clock::now();
  const double period = std::max(0.5, params_.depth_wall_tf_report_sec);
  if (std::chrono::duration<double>(now - depth_tf_report_last_).count() < period) {
    return;
  }
  depth_tf_report_last_ = now;
  std::ostringstream ss;
  bool any = false;
  for (size_t i = 0; i < depth_tf_total_.size(); ++i) {
    if (depth_tf_total_[i] == 0) {
      continue;
    }
    any = true;
    ss << ' ' << depth_cam_names_[i] << '=' << depth_tf_fail_[i] << '/' << depth_tf_total_[i] <<
      " (" << std::fixed << std::setprecision(1) <<
      100.0 * static_cast<double>(depth_tf_fail_[i]) / static_cast<double>(depth_tf_total_[i]) <<
      "%)";
    depth_tf_total_[i] = 0;
    depth_tf_fail_[i] = 0;
  }
  if (any) {
    RCLCPP_INFO(get_logger(), "depth TF fail rate:%s", ss.str().c_str());
  }
}

void WallPlaneFilterNode::wallFuseTick()
{
  CloudT walls;
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  rclcpp::Time stamp(0, 0, RCL_ROS_TIME);
  bool depth_update = false;
  {
    std::lock_guard<std::mutex> lock(depth_wall_mutex_);
    if (!wall_data_dirty_) {return;}
    const DepthSlot * newest = nullptr;
    for (const auto & slot : depth_slots_) {
      if (slot.fresh && (!newest || slot.stamp > newest->stamp)) {newest = &slot;}
    }
    if (newest) {
      depth_update = true;
      stamp = newest->stamp;
      center = newest->robot_xy;
      // A short synchronized snapshot, not a persistent map or extrapolated
      // track. Every final segment must pass this snapshot's support mask.
      for (auto & slot : depth_slots_) {
        if (slot.stamp.nanoseconds() != 0 &&
          std::abs((stamp - slot.stamp).seconds()) <= params_.depth_merge_max_age)
        {
          walls.insert(walls.end(), slot.wall_world.begin(), slot.wall_world.end());
        }
        slot.fresh = false;
      }
    }
    wall_data_dirty_ = false;
  }
  // Filtered clouds already went out in each sensor callback. This tick only
  // finds depth walls and refreshes depth band culling for display.
  if (depth_update) {
    HoughConfig hough_cfg = params_.depth_hough;
    {
      std::lock_guard<std::mutex> lock(depth_wall_mutex_);
      holdDepthGrid(center, hough_cfg);
    }
    const auto result = extractHoughWalls(walls, center, hough_cfg);
    std::vector<WallLine> tracked;
    {
      std::lock_guard<std::mutex> lock(depth_wall_mutex_);
      tracked = depth_tracks_.update(result.lines, stamp.seconds(), trackConfigFrom(params_.depth_hough));
      // Band culling reads these tracked map lines, not the raw Hough.
      validated_depth_lines_ = tracked;
      validated_depth_stamp_ = stamp;
    }
    if (params_.depth_hough.debug) {
      publishHoughDebug(result, stamp, wallWorkFrame(), true);
    }
    publishTrackedMarkers(tracked, result.corners, stamp, wallWorkFrame(), true);
    republishDepthClouds();
  }
}

void WallPlaneFilterNode::holdDepthGrid(const Eigen::Vector2d & robot, HoughConfig & config)
{
  if (!params_.depth_hough.follow_sensor) {
    return;
  }
  const double res = config.resolution;
  const int side = static_cast<int>(std::ceil(2.0 * config.half_extent / res));
  const double span = side * res;
  const double margin = std::max(0.5, 2.0 * res);
  const bool inside = depth_grid_valid_ &&
    robot.x() > depth_grid_origin_.x() + margin &&
    robot.y() > depth_grid_origin_.y() + margin &&
    robot.x() < depth_grid_origin_.x() + span - margin &&
    robot.y() < depth_grid_origin_.y() + span - margin;
  if (!inside) {
    depth_grid_origin_ = (robot / res).array().floor().matrix() * res -
      Eigen::Vector2d::Constant(span * 0.5);
    depth_grid_valid_ = true;
  }
  config.follow_sensor = false;
  config.origin_x = depth_grid_origin_.x();
  config.origin_y = depth_grid_origin_.y();
}

void WallPlaneFilterNode::republishDepthClouds()
{
  std::vector<DepthSlot> slots;
  {
    std::lock_guard<std::mutex> lock(depth_wall_mutex_);
    slots = depth_slots_;
  }
  for (size_t i = 0; i < slots.size(); ++i) {
    if (slots[i].stamp.nanoseconds() == 0) {
      continue;
    }
    publishDepthCloud(i, slots[i]);
  }
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

  visualization_msgs::msg::MarkerArray final;
  Marker clear;
  clear.header.frame_id = output;
  clear.header.stamp = stamp;
  clear.action = Marker::DELETEALL;
  final.markers.push_back(clear);

  wall_plane_filter::msg::WallPlaneArray planes;
  planes.header = clear.header;

  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  const bool tf_ok = output == frame || lookupIso(output, frame, stamp, transform);
  if (tf_ok && !lines.empty()) {
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
