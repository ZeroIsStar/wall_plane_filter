#include "wall_plane_filter/wall_hough.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <Eigen/Eigenvalues>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace wall_plane_filter
{
namespace
{
using Vec = Eigen::Vector2d;
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
constexpr double pi = 3.14159265358979323846;

WallLine segment(const Vec & a, const Vec & b)
{
  WallLine l;
  l.a = a;
  l.b = b;
  const Vec d = (b - a).normalized();
  l.normal = Vec(-d.y(), d.x());
  l.rho = l.normal.dot(a);
  return l;
}

double cross(const Vec & a, const Vec & b) {return a.x() * b.y() - a.y() * b.x();}

bool inBounds(const WallProjection & g, const Vec & p)
{
  const Vec q = (p - g.origin) / g.resolution;
  return q.x() >= 0 && q.y() >= 0 && q.x() < g.width && q.y() < g.height;
}

// Binary support is always measured on the original occupancy image, not a
// dilated detector image or Hough's own line pixels.
bool supported(const WallProjection & g, const Vec & p, double band)
{
  const int x = static_cast<int>(std::floor((p.x() - g.origin.x()) / g.resolution));
  const int y = static_cast<int>(std::floor((p.y() - g.origin.y()) / g.resolution));
  const int r = static_cast<int>(std::ceil(band / g.resolution));
  for (int yy = std::max(0, y - r); yy <= std::min(g.height - 1, y + r); ++yy) {
    for (int xx = std::max(0, x - r); xx <= std::min(g.width - 1, x + r); ++xx) {
      if (!g.binary[static_cast<size_t>(yy * g.width + xx)]) {continue;}
      const Vec q = g.origin + g.resolution * Vec(xx + 0.5, yy + 0.5);
      if ((q - p).squaredNorm() <= band * band) {return true;}
    }
  }
  return false;
}

// Grid associations retain the original point indices, so z extents and fit
// residuals are computed from real 3-D measurements rather than image pixels.
std::vector<size_t> supportIds(
  const WallLine & l, const Cloud & cloud, const WallProjection & g,
  const std::vector<std::vector<size_t>> & ids, double band)
{
  std::vector<size_t> result;
  const Vec d = (l.b - l.a).normalized();
  const double length = (l.b - l.a).norm();
  const Vec lo = l.a.cwiseMin(l.b) - Vec::Constant(band);
  const Vec hi = l.a.cwiseMax(l.b) + Vec::Constant(band);
  const int x0 = std::max(0, static_cast<int>(std::floor((lo.x() - g.origin.x()) / g.resolution)));
  const int y0 = std::max(0, static_cast<int>(std::floor((lo.y() - g.origin.y()) / g.resolution)));
  const int x1 = std::min(g.width - 1, static_cast<int>(std::floor((hi.x() - g.origin.x()) / g.resolution)));
  const int y1 = std::min(g.height - 1, static_cast<int>(std::floor((hi.y() - g.origin.y()) / g.resolution)));
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      const size_t cell = static_cast<size_t>(y * g.width + x);
      if (!g.binary[cell]) {continue;}
      for (size_t i : ids[cell]) {
        const Vec p(cloud[i].x, cloud[i].y);
        const double t = (p - l.a).dot(d);
        if (t >= -g.resolution * 0.5 && t <= length + g.resolution * 0.5 &&
          std::abs(l.normal.dot(p) - l.rho) <= band)
        {
          result.push_back(i);
        }
      }
    }
  }
  return result;
}

WallLine refineLine(
  const WallLine & l, const Cloud & cloud, const std::vector<size_t> & ids,
  const HoughConfig & c)
{
  if (!c.refine || ids.size() < 3) {return l;}
  Vec mean = Vec::Zero();
  for (size_t i : ids) {mean += Vec(cloud[i].x, cloud[i].y);}
  mean /= static_cast<double>(ids.size());
  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (size_t i : ids) {
    const Vec p = Vec(cloud[i].x, cloud[i].y) - mean;
    cov += p * p.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(cov);
  if (eig.info() != Eigen::Success) {return l;}
  const Vec d = eig.eigenvectors().col(1);
  if (std::abs(d.dot((l.b - l.a).normalized())) < std::cos(c.merge_angle_deg * pi / 180)) {
    return l;
  }
  const Vec a = mean + d * (l.a - mean).dot(d);
  const Vec b = mean + d * (l.b - mean).dot(d);
  return (b - a).norm() > 1e-6 ? segment(a, b) : l;
}

std::vector<WallLine> validateSegments(
  const WallLine & candidate, const Cloud & cloud, const WallProjection & g,
  const std::vector<std::vector<size_t>> & ids, const HoughConfig & c)
{
  WallLine l = refineLine(candidate, cloud, supportIds(candidate, cloud, g, ids, c.support_band), c);
  const double length = (l.b - l.a).norm();
  const Vec d = (l.b - l.a).normalized();
  const int intervals = std::max(1, static_cast<int>(std::ceil(length / (g.resolution * 0.5))));
  const double step = length / intervals;
  std::vector<int> hit;
  for (int i = 0; i <= intervals; ++i) {
    if (supported(g, l.a + d * (i * step), c.support_band)) {hit.push_back(i);}
  }
  std::vector<WallLine> result;
  for (size_t begin = 0; begin < hit.size();) {
    size_t end = begin + 1;
    while (end < hit.size() && (hit[end] - hit[end - 1] - 1) * step <= c.max_unsupported_gap + 1e-9) {++end;}
    const int first = hit[begin], last = hit[end - 1];
    const double ratio = static_cast<double>(end - begin) / (last - first + 1);
    int run = 1, longest = 1;
    for (size_t i = begin + 1; i < end; ++i) {
      run = hit[i] == hit[i - 1] + 1 ? run + 1 : 1;
      longest = std::max(longest, run);
    }
    if ((last - first) * step >= c.min_line_length && ratio >= c.min_support_ratio &&
      longest * step >= c.min_continuous_support)
    {
      WallLine part = segment(l.a + d * (first * step), l.a + d * (last * step));
      const auto points = supportIds(part, cloud, g, ids, c.support_band);
      if (points.size() >= static_cast<size_t>(c.min_support_points)) {
        part.z_min = std::numeric_limits<double>::infinity();
        part.z_max = -part.z_min;
        double error = 0;
        for (size_t i : points) {
          part.z_min = std::min(part.z_min, static_cast<double>(cloud[i].z));
          part.z_max = std::max(part.z_max, static_cast<double>(cloud[i].z));
          const double e = part.normal.dot(Vec(cloud[i].x, cloud[i].y)) - part.rho;
          error += e * e;
        }
        part.sigma = std::sqrt(error / points.size());
        part.support_points = points.size();
        part.support_ratio = ratio;
        result.push_back(part);
      }
    }
    begin = end;
  }
  return result;
}

bool merge(WallLine & a, const WallLine & b, const HoughConfig & c)
{
  const Vec d = (a.b - a.a).normalized();
  if (std::abs(d.dot((b.b - b.a).normalized())) < std::cos(c.merge_angle_deg * pi / 180)) {return false;}
  if (std::abs(a.normal.dot(b.a) - a.rho) > c.merge_distance ||
    std::abs(a.normal.dot(b.b) - a.rho) > c.merge_distance) {return false;}
  const double length = (a.b - a.a).norm();
  const double t0 = d.dot(b.a - a.a), t1 = d.dot(b.b - a.a);
  const double lo = std::min(t0, t1), hi = std::max(t0, t1);
  if (lo > length + c.merge_gap || hi < -c.merge_gap) {return false;}
  a = segment(a.a + d * std::min(0.0, lo), a.a + d * std::max(length, hi));
  return true;
}

bool duplicate(const WallLine & l, const std::vector<WallLine> & kept, const HoughConfig & c)
{
  for (const auto & k : kept) {
    const Vec d = (k.b - k.a).normalized();
    if (std::abs(d.dot((l.b - l.a).normalized())) < std::cos(c.merge_angle_deg * pi / 180)) {continue;}
    if (std::abs(k.normal.dot(l.a) - k.rho) > c.merge_distance ||
      std::abs(k.normal.dot(l.b) - k.rho) > c.merge_distance) {continue;}
    const double a = d.dot(l.a - k.a), b = d.dot(l.b - k.a);
    const double overlap = std::min((k.b - k.a).norm(), std::max(a, b)) - std::max(0.0, std::min(a, b));
    if (overlap >= 0.8 * std::min((l.b - l.a).norm(), (k.b - k.a).norm())) {return true;}
  }
  return false;
}
}  // namespace

void validateHoughConfig(const HoughConfig & c)
{
  const double values[] = {c.resolution, c.half_extent, c.min_cell_height, c.z_min, c.z_max,
    c.origin_x, c.origin_y, c.theta_deg, c.min_line_length, c.hough_max_gap, c.merge_angle_deg, c.merge_distance,
    c.merge_gap, c.parallel_angle_deg, c.intersection_extension, c.support_band,
    c.min_support_ratio, c.max_unsupported_gap, c.min_continuous_support};
  for (double v : values) {if (!std::isfinite(v)) {throw std::invalid_argument("Hough parameters must be finite");}}
  if (c.resolution < 0.01 || c.half_extent <= 0 || c.max_grid_side < 8 || c.max_grid_side > 1024 ||
    std::ceil(2 * c.half_extent / c.resolution) > c.max_grid_side || c.min_cell_points < 1 ||
    c.min_cell_height < 0 || c.z_max <= c.z_min || c.theta_deg < 0.1 || c.theta_deg > 10 ||
    c.votes < 1 || c.min_line_length < 2 * c.resolution || c.hough_max_gap < 0 ||
    c.merge_angle_deg <= 0 || c.merge_angle_deg >= 45 || c.merge_distance < 0 ||
    c.merge_gap < 0 || c.parallel_angle_deg <= 0 || c.parallel_angle_deg >= 90 ||
    c.intersection_extension < 0 || c.support_band < c.resolution * 0.71 ||
    c.support_band > std::min(c.half_extent, 10 * c.resolution) || c.min_support_ratio <= 0 || c.min_support_ratio > 1 ||
    c.max_unsupported_gap < 0 || c.min_continuous_support < 0 || c.min_support_points < 2 ||
    c.max_lines < 1 || c.max_lines > 256)
  {
    throw std::invalid_argument("Invalid Hough settings: check grid limit, lengths, angles and support thresholds");
  }
}

HoughResult extractHoughWalls(const Cloud & cloud, const Vec & center, const HoughConfig & c)
{
  validateHoughConfig(c);
  if (!center.allFinite()) {throw std::invalid_argument("Hough grid center is not finite");}
  const auto start = std::chrono::steady_clock::now();
  HoughResult out;
  auto & g = out.grid;
  g.resolution = c.resolution;
  g.width = g.height = static_cast<int>(std::ceil(2 * c.half_extent / c.resolution));
  g.origin = (center / c.resolution).array().floor().matrix() * c.resolution - Vec::Constant(g.width * c.resolution * 0.5);
  if (!c.follow_sensor) {g.origin = Vec(c.origin_x, c.origin_y);}
  const size_t cells = static_cast<size_t>(g.width) * g.height;
  g.counts.assign(cells, 0);
  g.binary.assign(cells, 0);
  std::vector<std::vector<size_t>> ids(cells);
  std::vector<double> zlo(cells, std::numeric_limits<double>::infinity()), zhi(cells, -std::numeric_limits<double>::infinity());
  out.input_points = cloud.size();
  for (size_t i = 0; i < cloud.size(); ++i) {
    const auto & p = cloud[i];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {continue;}
    const Vec xy(p.x, p.y);
    if (!inBounds(g, xy)) {continue;}
    const int x = static_cast<int>(std::floor((p.x - g.origin.x()) / c.resolution));
    const int y = static_cast<int>(std::floor((p.y - g.origin.y()) / c.resolution));
    const size_t cell = static_cast<size_t>(y * g.width + x);
    ++g.counts[cell];
    ids[cell].push_back(i);
    zlo[cell] = std::min(zlo[cell], static_cast<double>(p.z));
    zhi[cell] = std::max(zhi[cell], static_cast<double>(p.z));
    ++out.projected_points;
  }
  for (size_t i = 0; i < cells; ++i) {
    if (g.counts[i] >= static_cast<uint32_t>(c.min_cell_points) && zhi[i] - zlo[i] >= c.min_cell_height) {
      g.binary[i] = 255;
      ++out.occupied_cells;
    }
  }
  cv::Mat image(g.height, g.width, CV_8UC1, g.binary.data());
  std::vector<cv::Vec4i> detected;
  // No closing/dilation: doors remain absent in the validation mask.
  cv::HoughLinesP(image.clone(), detected, 1.0, c.theta_deg * pi / 180, c.votes,
    c.min_line_length / c.resolution, c.hough_max_gap / c.resolution);
  std::sort(detected.begin(), detected.end(), [](const cv::Vec4i & a, const cv::Vec4i & b) {
    return std::hypot(a[2] - a[0], a[3] - a[1]) > std::hypot(b[2] - b[0], b[3] - b[1]);
  });
  if (detected.size() > static_cast<size_t>(c.max_lines)) {detected.resize(c.max_lines);}
  std::vector<WallLine> merged;
  for (const auto & v : detected) {
    WallLine line = segment(g.origin + c.resolution * Vec(v[0] + .5, v[1] + .5),
      g.origin + c.resolution * Vec(v[2] + .5, v[3] + .5));
    out.hough_lines.push_back(line);
    bool found = false;
    for (auto & previous : merged) {if (merge(previous, line, c)) {found = true; break;}}
    if (!found) {merged.push_back(line);}
  }
  std::vector<std::vector<double>> cuts(merged.size());
  for (size_t i = 0; i < merged.size(); ++i) {cuts[i] = {0, (merged[i].b - merged[i].a).norm()};}
  for (size_t i = 0; i < merged.size(); ++i) {
    for (size_t j = i + 1; j < merged.size(); ++j) {
      const Vec d = (merged[i].b - merged[i].a).normalized(), e = (merged[j].b - merged[j].a).normalized();
      const double det = cross(d, e);
      if (std::abs(det) < std::sin(c.parallel_angle_deg * pi / 180)) {continue;}
      const Vec offset = merged[j].a - merged[i].a;
      const double t = cross(offset, e) / det, u = cross(offset, d) / det;
      if (t < -c.intersection_extension || t > (merged[i].b - merged[i].a).norm() + c.intersection_extension ||
        u < -c.intersection_extension || u > (merged[j].b - merged[j].a).norm() + c.intersection_extension) {continue;}
      const Vec p = merged[i].a + t * d;
      if (!inBounds(g, p)) {continue;}
      out.intersections.push_back(p);
      cuts[i].push_back(t);
      cuts[j].push_back(u);
    }
  }
  for (size_t i = 0; i < merged.size(); ++i) {
    auto & ts = cuts[i];
    const double length = (merged[i].b - merged[i].a).norm();
    const double lo = *std::min_element(ts.begin(), ts.end()), hi = *std::max_element(ts.begin(), ts.end());
    // Replace Hough endpoints when a nearby corner extends the line. Do not
    // retain tiny isolated extension segments that cannot pass min length.
    ts.erase(std::remove_if(ts.begin(), ts.end(), [&](double t) {
      return (lo < 0 && t == 0) || (hi > length && t == length);
    }), ts.end());
    std::sort(ts.begin(), ts.end());
    ts.erase(std::unique(ts.begin(), ts.end(), [](double a, double b) {return std::abs(a - b) < 1e-6;}), ts.end());
    const Vec d = (merged[i].b - merged[i].a).normalized();
    for (size_t k = 1; k < ts.size(); ++k) {
      if (ts[k] - ts[k - 1] < c.min_line_length) {continue;}
      const auto candidate = segment(merged[i].a + ts[k - 1] * d, merged[i].a + ts[k] * d);
      out.candidates.push_back(candidate);
      for (const auto & line : validateSegments(candidate, cloud, g, ids, c)) {
        if (!duplicate(line, out.lines, c)) {out.lines.push_back(line);}
      }
    }
  }
  for (const auto & p : out.intersections) {
    std::vector<Vec> directions;
    for (const auto & l : out.lines) {
      if (std::min((l.a - p).norm(), (l.b - p).norm()) <= c.support_band + c.resolution) {
        directions.push_back((l.b - l.a).normalized());
      }
    }
    bool corner = false;
    for (size_t i = 0; i < directions.size(); ++i) {
      for (size_t j = i + 1; j < directions.size(); ++j) {
        corner |= std::abs(cross(directions[i], directions[j])) >= std::sin(c.parallel_angle_deg * pi / 180);
      }
    }
    if (corner && std::none_of(out.corners.begin(), out.corners.end(), [&](const Vec & q) {return (p - q).norm() < c.resolution;})) {
      out.corners.push_back(p);
    }
  }
  out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return out;
}
}  // namespace wall_plane_filter
