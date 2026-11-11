#include "wall_plane_filter/wall_hough.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

bool merge(WallLine & a, const WallLine & b, const HoughConfig & c)
{
  const Vec d = (a.b - a.a).normalized();
  if (std::abs(d.dot((b.b - b.a).normalized())) < std::cos(c.merge_angle_deg * pi / 180)) {
    return false;
  }
  if (std::abs(a.normal.dot(b.a) - a.rho) > c.merge_distance ||
    std::abs(a.normal.dot(b.b) - a.rho) > c.merge_distance)
  {
    return false;
  }
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
    if (std::abs(d.dot((l.b - l.a).normalized())) < std::cos(c.merge_angle_deg * pi / 180)) {
      continue;
    }
    if (std::abs(k.normal.dot(l.a) - k.rho) > c.merge_distance ||
      std::abs(k.normal.dot(l.b) - k.rho) > c.merge_distance)
    {
      continue;
    }
    const double a = d.dot(l.a - k.a), b = d.dot(l.b - k.a);
    const double overlap =
      std::min((k.b - k.a).norm(), std::max(a, b)) - std::max(0.0, std::min(a, b));
    if (overlap >= 0.8 * std::min((l.b - l.a).norm(), (k.b - k.a).norm())) {return true;}
  }
  return false;
}

std::string normalizeExtractor(std::string name)
{
  for (char & ch : name) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return name;
}

bool isRansacExtractor(const HoughConfig & c)
{
  return normalizeExtractor(c.extractor) == "ransac";
}

bool isSeedGrowExtractor(const HoughConfig & c)
{
  return normalizeExtractor(c.extractor) == "seed_grow";
}

double ransacDist(const HoughConfig & c)
{
  return (c.ransac_distance > 1e-9) ? c.ransac_distance : c.support_band;
}

double seedDist(const HoughConfig & c)
{
  return (c.seed_max_distance > 1e-9) ? c.seed_max_distance : c.support_band;
}

double growDist(const HoughConfig & c)
{
  return (c.grow_distance > 1e-9) ? c.grow_distance : c.support_band;
}

struct XySample
{
  Vec xy{Vec::Zero()};
  double z{0};
  size_t src{0};
};

std::vector<XySample> downsampleXy(
  const Cloud & cloud, const Vec & center, double half_extent, double leaf)
{
  std::vector<XySample> out;
  out.reserve(cloud.size());
  const double ext2 = half_extent * half_extent;
  if (!(leaf > 1e-6)) {
    for (size_t i = 0; i < cloud.size(); ++i) {
      const auto & p = cloud[i];
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {continue;}
      const Vec xy(p.x, p.y);
      if ((xy - center).squaredNorm() > ext2) {continue;}
      out.push_back({xy, static_cast<double>(p.z), i});
    }
    return out;
  }
  std::unordered_map<uint64_t, size_t> occupied;
  occupied.reserve(cloud.size() / 2 + 1);
  const double inv = 1.0 / leaf;
  for (size_t i = 0; i < cloud.size(); ++i) {
    const auto & p = cloud[i];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {continue;}
    const Vec xy(p.x, p.y);
    if ((xy - center).squaredNorm() > ext2) {continue;}
    const int64_t ix = static_cast<int64_t>(std::floor(p.x * inv));
    const int64_t iy = static_cast<int64_t>(std::floor(p.y * inv));
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(ix)) << 32) |
      static_cast<uint32_t>(iy);
    if (!occupied.emplace(key, out.size()).second) {continue;}
    out.push_back({xy, static_cast<double>(p.z), i});
  }
  return out;
}

WallLine refineLineSamples(
  const WallLine & seed, const std::vector<XySample> & samples,
  const std::vector<size_t> & ids, const HoughConfig & c)
{
  if (!c.refine || ids.size() < 3) {return seed;}
  Vec mean = Vec::Zero();
  for (size_t i : ids) {mean += samples[i].xy;}
  mean /= static_cast<double>(ids.size());
  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (size_t i : ids) {
    const Vec p = samples[i].xy - mean;
    cov += p * p.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(cov);
  if (eig.info() != Eigen::Success) {return seed;}
  const Vec d = eig.eigenvectors().col(1);
  if (d.squaredNorm() < 1e-12) {return seed;}
  const Vec dir = d.normalized();
  if (std::abs(dir.dot((seed.b - seed.a).normalized())) < std::cos(c.merge_angle_deg * pi / 180)) {
    return seed;
  }
  double t_min = std::numeric_limits<double>::infinity();
  double t_max = -t_min;
  for (size_t i : ids) {
    const double t = (samples[i].xy - mean).dot(dir);
    t_min = std::min(t_min, t);
    t_max = std::max(t_max, t);
  }
  if (!(t_max - t_min > 1e-6)) {return seed;}
  return segment(mean + dir * t_min, mean + dir * t_max);
}

std::vector<WallLine> validateFromSamples(
  const WallLine & candidate, const std::vector<XySample> & samples,
  const std::vector<size_t> & pool, const HoughConfig & c, double band)
{
  const Vec d0 = (candidate.b - candidate.a).normalized();
  const double len0 = (candidate.b - candidate.a).norm();
  std::vector<size_t> near;
  near.reserve(pool.size());
  for (size_t i : pool) {
    const Vec & p = samples[i].xy;
    const double t = (p - candidate.a).dot(d0);
    if (t < -band || t > len0 + band) {continue;}
    if (std::abs(candidate.normal.dot(p) - candidate.rho) > band) {continue;}
    near.push_back(i);
  }
  WallLine l = refineLineSamples(candidate, samples, near, c);
  const double length = (l.b - l.a).norm();
  if (length < c.min_line_length) {return {};}
  const Vec d = (l.b - l.a).normalized();
  struct Hit {double t; size_t id;};
  std::vector<Hit> hits;
  hits.reserve(near.size());
  for (size_t i : near) {
    const Vec & p = samples[i].xy;
    const double t = (p - l.a).dot(d);
    if (t < -band || t > length + band) {continue;}
    if (std::abs(l.normal.dot(p) - l.rho) > band) {continue;}
    hits.push_back({t, i});
  }
  if (hits.size() < static_cast<size_t>(c.min_support_points)) {return {};}
  std::sort(hits.begin(), hits.end(), [](const Hit & a, const Hit & b) {return a.t < b.t;});

  std::vector<WallLine> result;
  for (size_t begin = 0; begin < hits.size();) {
    size_t end = begin + 1;
    while (end < hits.size() &&
      (hits[end].t - hits[end - 1].t) <= c.max_unsupported_gap + 1e-9)
    {
      ++end;
    }
    const double t0 = hits[begin].t;
    const double t1 = hits[end - 1].t;
    const double span = t1 - t0;
    const size_t n = end - begin;
    if (span < c.min_line_length || n < static_cast<size_t>(c.min_support_points) ||
      span < c.min_continuous_support)
    {
      begin = end;
      continue;
    }
    const double ratio =
      std::min(1.0, static_cast<double>(n) * c.max_unsupported_gap / std::max(span, 1e-6));
    if (ratio < c.min_support_ratio) {
      begin = end;
      continue;
    }
    WallLine part = segment(l.a + d * t0, l.a + d * t1);
    part.z_min = std::numeric_limits<double>::infinity();
    part.z_max = -part.z_min;
    double err = 0;
    for (size_t k = begin; k < end; ++k) {
      const auto & s = samples[hits[k].id];
      part.z_min = std::min(part.z_min, s.z);
      part.z_max = std::max(part.z_max, s.z);
      const double e = part.normal.dot(s.xy) - part.rho;
      err += e * e;
    }
    part.sigma = std::sqrt(err / static_cast<double>(n));
    part.support_points = n;
    part.support_ratio = ratio;
    result.push_back(part);
    begin = end;
  }
  return result;
}

bool fitOneRansacLine(
  const std::vector<XySample> & samples, const std::vector<size_t> & active,
  const HoughConfig & c, std::mt19937 & rng, WallLine & out_line,
  std::vector<size_t> & out_inliers)
{
  out_inliers.clear();
  if (active.size() < static_cast<size_t>(std::max(2, c.ransac_min_inliers))) {return false;}
  const double dist = ransacDist(c);
  std::uniform_int_distribution<size_t> pick(0, active.size() - 1);
  size_t best_count = 0;
  WallLine best_seed;
  std::vector<size_t> best_ids;
  best_ids.reserve(active.size());

  for (int iter = 0; iter < c.ransac_max_iterations; ++iter) {
    const size_t i0 = active[pick(rng)];
    size_t i1 = active[pick(rng)];
    if (i1 == i0) {
      i1 = active[(pick(rng) + 1) % active.size()];
    }
    if ((samples[i1].xy - samples[i0].xy).squaredNorm() < 1e-4) {continue;}
    const WallLine seed = segment(samples[i0].xy, samples[i1].xy);
    std::vector<size_t> ids;
    ids.reserve(active.size());
    double t_min = std::numeric_limits<double>::infinity();
    double t_max = -t_min;
    const Vec d = (seed.b - seed.a).normalized();
    for (size_t idx : active) {
      const Vec & p = samples[idx].xy;
      if (std::abs(seed.normal.dot(p) - seed.rho) > dist) {continue;}
      const double t = (p - seed.a).dot(d);
      t_min = std::min(t_min, t);
      t_max = std::max(t_max, t);
      ids.push_back(idx);
    }
    if (ids.size() < static_cast<size_t>(c.ransac_min_inliers)) {continue;}
    if (!(t_max - t_min >= c.min_line_length)) {continue;}
    if (ids.size() > best_count) {
      best_count = ids.size();
      best_seed = segment(seed.a + d * t_min, seed.a + d * t_max);
      best_ids.swap(ids);
    }
  }
  if (best_count < static_cast<size_t>(c.ransac_min_inliers)) {return false;}
  out_line = refineLineSamples(best_seed, samples, best_ids, c);
  if ((out_line.b - out_line.a).norm() < c.min_line_length) {return false;}
  const double band = std::max(dist, c.support_band);
  const Vec d = (out_line.b - out_line.a).normalized();
  const double length = (out_line.b - out_line.a).norm();
  out_inliers.clear();
  for (size_t idx : active) {
    const Vec & p = samples[idx].xy;
    const double t = (p - out_line.a).dot(d);
    if (t < -band || t > length + band) {continue;}
    if (std::abs(out_line.normal.dot(p) - out_line.rho) > band) {continue;}
    out_inliers.push_back(idx);
  }
  return out_inliers.size() >= static_cast<size_t>(c.ransac_min_inliers);
}

void fillDebugGridFromSamples(
  HoughResult & out, const std::vector<XySample> & samples, const Vec & center,
  const HoughConfig & c)
{
  auto & g = out.grid;
  g.resolution = c.resolution;
  g.width = g.height = static_cast<int>(std::ceil(2 * c.half_extent / c.resolution));
  if (c.max_grid_side > 0) {
    g.width = std::min(g.width, c.max_grid_side);
    g.height = std::min(g.height, c.max_grid_side);
  }
  g.origin = (center / c.resolution).array().floor().matrix() * c.resolution -
    Vec::Constant(g.width * c.resolution * 0.5);
  if (!c.follow_sensor) {g.origin = Vec(c.origin_x, c.origin_y);}
  const size_t cells = static_cast<size_t>(g.width) * g.height;
  g.counts.assign(cells, 0);
  g.binary.assign(cells, 0);
  for (const auto & s : samples) {
    if (!inBounds(g, s.xy)) {continue;}
    const int x = static_cast<int>(std::floor((s.xy.x() - g.origin.x()) / g.resolution));
    const int y = static_cast<int>(std::floor((s.xy.y() - g.origin.y()) / g.resolution));
    const size_t cell = static_cast<size_t>(y * g.width + x);
    ++g.counts[cell];
    ++out.projected_points;
  }
  for (size_t i = 0; i < cells; ++i) {
    if (g.counts[i] >= static_cast<uint32_t>(std::max(1, c.min_cell_points))) {
      g.binary[i] = 255;
      ++out.occupied_cells;
    }
  }
}

void appendCornerHints(HoughResult & out, const HoughConfig & c)
{
  for (size_t i = 0; i < out.lines.size(); ++i) {
    for (size_t j = i + 1; j < out.lines.size(); ++j) {
      const WallLine & li = out.lines[i];
      const WallLine & lj = out.lines[j];
      const double len_i = (li.b - li.a).norm();
      const double len_j = (lj.b - lj.a).norm();
      if (len_i < 1e-6 || len_j < 1e-6) {continue;}
      const Vec di = (li.b - li.a) / len_i;
      const Vec dj = (lj.b - lj.a) / len_j;
      const double det = cross(di, dj);
      if (std::abs(det) < std::sin(c.parallel_angle_deg * pi / 180)) {continue;}
      const Vec offset = lj.a - li.a;
      const double t = cross(offset, dj) / det;
      const double u = cross(offset, di) / det;
      if (t < -c.intersection_extension || t > len_i + c.intersection_extension ||
        u < -c.intersection_extension || u > len_j + c.intersection_extension)
      {
        continue;
      }
      const Vec p = li.a + t * di;
      out.intersections.push_back(p);
      const double end_i = std::min(std::abs(t), std::abs(t - len_i));
      const double end_j = std::min(std::abs(u), std::abs(u - len_j));
      if (end_i <= c.intersection_extension && end_j <= c.intersection_extension) {
        if (std::none_of(out.corners.begin(), out.corners.end(), [&](const Vec & q) {
            return (p - q).norm() < std::max(0.05, c.resolution);
          }))
        {
          out.corners.push_back(p);
        }
      }
    }
  }
}

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
  if (length < c.min_line_length) {return {};}
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
    while (end < hit.size() &&
      (hit[end] - hit[end - 1] - 1) * step <= c.max_unsupported_gap + 1e-9)
    {
      ++end;
    }
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

HoughResult extractRansacWalls(const Cloud & cloud, const Vec & center, const HoughConfig & c)
{
  const auto start = std::chrono::steady_clock::now();
  HoughResult out;
  out.extractor = "ransac";
  out.input_points = cloud.size();
  auto samples = downsampleXy(cloud, center, c.half_extent, c.ransac_downsample_leaf);
  if (c.debug) {
    fillDebugGridFromSamples(out, samples, center, c);
  } else {
    out.grid.resolution = c.resolution;
    out.grid.width = out.grid.height = 0;
  }

  std::vector<size_t> active(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {active[i] = i;}
  std::mt19937 rng(static_cast<uint32_t>(samples.size() * 1315423911u + 17u));

  std::vector<WallLine> raw;
  while (static_cast<int>(raw.size()) < c.max_lines &&
    active.size() >= static_cast<size_t>(std::max(2, c.ransac_min_inliers)))
  {
    WallLine line;
    std::vector<size_t> inliers;
    if (!fitOneRansacLine(samples, active, c, rng, line, inliers)) {break;}
    raw.push_back(line);
    out.hough_lines.push_back(line);
    std::vector<char> drop(samples.size(), 0);
    for (size_t id : inliers) {drop[id] = 1;}
    std::vector<size_t> next;
    next.reserve(active.size() - inliers.size());
    for (size_t id : active) {
      if (!drop[id]) {next.push_back(id);}
    }
    active.swap(next);
  }

  std::vector<WallLine> merged;
  for (const auto & line : raw) {
    bool found = false;
    for (auto & previous : merged) {
      if (merge(previous, line, c)) {found = true; break;}
    }
    if (!found) {merged.push_back(line);}
  }

  std::vector<size_t> pool(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {pool[i] = i;}
  const double band = std::max(ransacDist(c), c.support_band);
  for (const auto & cand : merged) {
    out.candidates.push_back(cand);
    for (const auto & line : validateFromSamples(cand, samples, pool, c, band)) {
      if (!duplicate(line, out.lines, c)) {out.lines.push_back(line);}
    }
  }
  appendCornerHints(out, c);
  out.elapsed_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
  return out;
}

HoughResult extractHoughWallsGrid(const Cloud & cloud, const Vec & center, const HoughConfig & c)
{
  const auto start = std::chrono::steady_clock::now();
  HoughResult out;
  out.extractor = "hough";
  auto & g = out.grid;
  g.resolution = c.resolution;
  g.width = g.height = static_cast<int>(std::ceil(2 * c.half_extent / c.resolution));
  if (c.max_grid_side > 0) {
    g.width = std::min(g.width, c.max_grid_side);
    g.height = std::min(g.height, c.max_grid_side);
  }
  g.origin = (center / c.resolution).array().floor().matrix() * c.resolution -
    Vec::Constant(g.width * c.resolution * 0.5);
  if (!c.follow_sensor) {g.origin = Vec(c.origin_x, c.origin_y);}
  const size_t cells = static_cast<size_t>(g.width) * g.height;
  g.counts.assign(cells, 0);
  g.binary.assign(cells, 0);
  std::vector<std::vector<size_t>> ids(cells);
  std::vector<double> zlo(cells, std::numeric_limits<double>::infinity());
  std::vector<double> zhi(cells, -std::numeric_limits<double>::infinity());
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
    if (g.counts[i] >= static_cast<uint32_t>(c.min_cell_points) &&
      zhi[i] - zlo[i] >= c.min_cell_height)
    {
      g.binary[i] = 255;
      ++out.occupied_cells;
    }
  }
  cv::Mat image(g.height, g.width, CV_8UC1, g.binary.data());
  std::vector<cv::Vec4i> detected;
  cv::HoughLinesP(
    image.clone(), detected, 1.0, c.theta_deg * pi / 180, c.votes,
    c.min_line_length / c.resolution, c.hough_max_gap / c.resolution);
  std::sort(detected.begin(), detected.end(), [](const cv::Vec4i & a, const cv::Vec4i & b) {
    return std::hypot(a[2] - a[0], a[3] - a[1]) > std::hypot(b[2] - b[0], b[3] - b[1]);
  });
  if (detected.size() > static_cast<size_t>(c.max_lines)) {detected.resize(c.max_lines);}
  std::vector<WallLine> merged;
  for (const auto & v : detected) {
    WallLine line = segment(
      g.origin + c.resolution * Vec(v[0] + 0.5, v[1] + 0.5),
      g.origin + c.resolution * Vec(v[2] + 0.5, v[3] + 0.5));
    out.hough_lines.push_back(line);
    bool found = false;
    for (auto & previous : merged) {
      if (merge(previous, line, c)) {found = true; break;}
    }
    if (!found) {merged.push_back(line);}
  }
  std::vector<std::vector<double>> cuts(merged.size());
  for (size_t i = 0; i < merged.size(); ++i) {
    cuts[i] = {0, (merged[i].b - merged[i].a).norm()};
  }
  for (size_t i = 0; i < merged.size(); ++i) {
    for (size_t j = i + 1; j < merged.size(); ++j) {
      const Vec d = (merged[i].b - merged[i].a).normalized();
      const Vec e = (merged[j].b - merged[j].a).normalized();
      const double det = cross(d, e);
      if (std::abs(det) < std::sin(c.parallel_angle_deg * pi / 180)) {continue;}
      const Vec offset = merged[j].a - merged[i].a;
      const double t = cross(offset, e) / det;
      const double u = cross(offset, d) / det;
      if (t < -c.intersection_extension ||
        t > (merged[i].b - merged[i].a).norm() + c.intersection_extension ||
        u < -c.intersection_extension ||
        u > (merged[j].b - merged[j].a).norm() + c.intersection_extension)
      {
        continue;
      }
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
    const double lo = *std::min_element(ts.begin(), ts.end());
    const double hi = *std::max_element(ts.begin(), ts.end());
    ts.erase(std::remove_if(ts.begin(), ts.end(), [&](double t) {
        return (lo < 0 && t == 0) || (hi > length && t == length);
      }), ts.end());
    std::sort(ts.begin(), ts.end());
    ts.erase(std::unique(ts.begin(), ts.end(), [](double a, double b) {
        return std::abs(a - b) < 1e-6;
      }), ts.end());
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
  appendCornerHints(out, c);
  out.elapsed_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
  return out;
}

struct ContourBin
{
  bool valid{false};
  Vec xy{Vec::Zero()};
  double z{0};
  double range{0};
  size_t src{0};
};

std::vector<ContourBin> buildPolarContour(
  const Cloud & cloud, const Vec & center, const HoughConfig & c)
{
  const double ang = std::max(0.1, c.seed_contour_angle_deg) * pi / 180.0;
  const int n_bins = std::max(8, static_cast<int>(std::ceil(2.0 * pi / ang)));
  std::vector<ContourBin> bins(static_cast<size_t>(n_bins));
  const double ext2 = c.half_extent * c.half_extent;
  for (size_t i = 0; i < cloud.size(); ++i) {
    const auto & p = cloud[i];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {continue;}
    const Vec xy(p.x, p.y);
    const Vec d = xy - center;
    const double r2 = d.squaredNorm();
    if (!(r2 > 1e-8) || r2 > ext2) {continue;}
    double yaw = std::atan2(d.y(), d.x());
    if (yaw < 0.0) {yaw += 2.0 * pi;}
    int bin = static_cast<int>(yaw / ang);
    if (bin < 0) {bin = 0;}
    if (bin >= n_bins) {bin = n_bins - 1;}
    ContourBin & slot = bins[static_cast<size_t>(bin)];
    const double r = std::sqrt(r2);
    if (!slot.valid || r < slot.range) {
      slot.valid = true;
      slot.xy = xy;
      slot.z = static_cast<double>(p.z);
      slot.range = r;
      slot.src = i;
    }
  }
  return bins;
}

bool fitLineToBins(
  const std::vector<ContourBin> & bins, int lo, int hi, WallLine & line, double * max_err)
{
  std::vector<size_t> ids;
  Vec mean = Vec::Zero();
  for (int i = lo; i <= hi; ++i) {
    if (i < 0 || i >= static_cast<int>(bins.size()) || !bins[static_cast<size_t>(i)].valid) {
      continue;
    }
    ids.push_back(static_cast<size_t>(i));
    mean += bins[static_cast<size_t>(i)].xy;
  }
  if (ids.size() < 2) {return false;}
  mean /= static_cast<double>(ids.size());
  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (size_t id : ids) {
    const Vec p = bins[id].xy - mean;
    cov += p * p.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(cov);
  if (eig.info() != Eigen::Success) {return false;}
  const Vec dir = eig.eigenvectors().col(1).normalized();
  if (dir.squaredNorm() < 1e-12) {return false;}
  double t_min = std::numeric_limits<double>::infinity();
  double t_max = -t_min;
  double err = 0.0;
  for (size_t id : ids) {
    const Vec p = bins[id].xy - mean;
    const double t = p.dot(dir);
    t_min = std::min(t_min, t);
    t_max = std::max(t_max, t);
    err = std::max(err, std::abs(p.dot(Vec(-dir.y(), dir.x()))));
  }
  if (!(t_max - t_min > 1e-6)) {return false;}
  line = segment(mean + dir * t_min, mean + dir * t_max);
  if (max_err) {*max_err = err;}
  return true;
}

double pointLineDist(const WallLine & line, const Vec & p)
{
  return std::abs(line.normal.dot(p) - line.rho);
}

HoughResult extractSeedGrowWalls(const Cloud & cloud, const Vec & center, const HoughConfig & c)
{
  const auto start = std::chrono::steady_clock::now();
  HoughResult out;
  out.extractor = "seed_grow";
  out.input_points = cloud.size();

  auto bins = buildPolarContour(cloud, center, c);
  size_t valid_n = 0;
  for (const auto & b : bins) {if (b.valid) {++valid_n;}}
  out.projected_points = valid_n;

  std::vector<XySample> samples;
  samples.reserve(valid_n);
  for (const auto & b : bins) {
    if (b.valid) {samples.push_back({b.xy, b.z, b.src});}
  }
  if (c.debug) {
    fillDebugGridFromSamples(out, samples, center, c);
  } else {
    out.grid.resolution = c.resolution;
    out.grid.width = out.grid.height = 0;
  }

  const int n = static_cast<int>(bins.size());
  const int seed_n = std::max(3, c.seed_min_points);
  const double seed_th = seedDist(c);
  const double grow_th = growDist(c);
  const int gap_bins = std::max(0, c.grow_max_gap_bins);
  const double jump = std::max(0.0, c.grow_max_range_jump);
  std::vector<char> used(static_cast<size_t>(n), 0);
  auto bin_ok = [&](int i) {
    return i >= 0 && i < n && bins[static_cast<size_t>(i)].valid && !used[static_cast<size_t>(i)];
  };

  std::vector<WallLine> grown;
  for (int start_i = 0; start_i < n && static_cast<int>(grown.size()) < c.max_lines; ) {
    if (!bin_ok(start_i)) {++start_i; continue;}
    std::vector<int> seed_idx;
    int cursor = start_i;
    int gaps = 0;
    while (cursor < n && static_cast<int>(seed_idx.size()) < seed_n) {
      if (bin_ok(cursor)) {
        if (!seed_idx.empty()) {
          const auto & prev = bins[static_cast<size_t>(seed_idx.back())];
          const auto & cur = bins[static_cast<size_t>(cursor)];
          if (jump > 1e-9 && std::abs(cur.range - prev.range) > jump) {break;}
        }
        seed_idx.push_back(cursor);
        gaps = 0;
      } else {
        ++gaps;
        if (gaps > gap_bins) {break;}
      }
      ++cursor;
    }
    if (static_cast<int>(seed_idx.size()) < seed_n) {++start_i; continue;}

    WallLine seed;
    double seed_err = 0.0;
    if (!fitLineToBins(bins, seed_idx.front(), seed_idx.back(), seed, &seed_err) ||
      seed_err > seed_th || (seed.b - seed.a).norm() < c.min_line_length * 0.35)
    {
      ++start_i;
      continue;
    }
    out.hough_lines.push_back(seed);

    int left = seed_idx.front();
    int right = seed_idx.back();
    int miss = 0;
    for (int i = left - 1; i >= 0; --i) {
      if (!bin_ok(i)) {
        ++miss;
        if (miss > gap_bins) {break;}
        continue;
      }
      miss = 0;
      const auto & b = bins[static_cast<size_t>(i)];
      const auto & nbr = bins[static_cast<size_t>(left)];
      if (jump > 1e-9 && std::abs(b.range - nbr.range) > jump) {break;}
      if (pointLineDist(seed, b.xy) > grow_th) {break;}
      left = i;
      fitLineToBins(bins, left, right, seed, nullptr);
    }
    miss = 0;
    for (int i = right + 1; i < n; ++i) {
      if (!bin_ok(i)) {
        ++miss;
        if (miss > gap_bins) {break;}
        continue;
      }
      miss = 0;
      const auto & b = bins[static_cast<size_t>(i)];
      const auto & nbr = bins[static_cast<size_t>(right)];
      if (jump > 1e-9 && std::abs(b.range - nbr.range) > jump) {break;}
      if (pointLineDist(seed, b.xy) > grow_th) {break;}
      right = i;
      fitLineToBins(bins, left, right, seed, nullptr);
    }

    if ((seed.b - seed.a).norm() < c.min_line_length) {++start_i; continue;}
    for (int i = left; i <= right; ++i) {
      if (i >= 0 && i < n && bins[static_cast<size_t>(i)].valid) {
        used[static_cast<size_t>(i)] = 1;
      }
    }
    grown.push_back(seed);
    out.candidates.push_back(seed);
    start_i = right + 1;
  }

  std::vector<WallLine> merged;
  for (const auto & line : grown) {
    bool found = false;
    for (auto & previous : merged) {
      if (merge(previous, line, c)) {found = true; break;}
    }
    if (!found) {merged.push_back(line);}
  }

  std::vector<size_t> pool(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {pool[i] = i;}
  const double band = std::max(grow_th, c.support_band);
  for (const auto & cand : merged) {
    for (const auto & line : validateFromSamples(cand, samples, pool, c, band)) {
      if (!duplicate(line, out.lines, c)) {out.lines.push_back(line);}
    }
  }
  out.elapsed_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
  return out;
}
}  // namespace

void validateHoughConfig(const HoughConfig & c)
{
  const std::string mode = normalizeExtractor(c.extractor);
  if (mode != "hough" && mode != "ransac" && mode != "seed_grow") {
    throw std::invalid_argument(
      "HoughConfig.extractor must be \"hough\", \"ransac\", or \"seed_grow\"");
  }
  const double shared[] = {
    c.resolution, c.half_extent, c.z_min, c.z_max, c.origin_x, c.origin_y,
    c.min_line_length, c.merge_angle_deg, c.merge_distance, c.merge_gap,
    c.parallel_angle_deg, c.intersection_extension, c.support_band,
    c.min_support_ratio, c.max_unsupported_gap, c.min_continuous_support,
    c.ransac_distance, c.ransac_downsample_leaf, c.theta_deg, c.hough_max_gap,
    c.min_cell_height, c.seed_contour_angle_deg, c.seed_max_distance,
    c.grow_distance, c.grow_max_range_jump};
  for (double v : shared) {
    if (!std::isfinite(v)) {
      throw std::invalid_argument("Wall extractor parameters must be finite");
    }
  }
  if (c.half_extent <= 0 || c.z_max <= c.z_min || c.min_line_length <= 0 ||
    c.merge_angle_deg <= 0 || c.merge_angle_deg >= 45 || c.merge_distance < 0 ||
    c.merge_gap < 0 || c.parallel_angle_deg <= 0 || c.parallel_angle_deg >= 90 ||
    c.intersection_extension < 0 || c.support_band <= 0 ||
    c.min_support_ratio <= 0 || c.min_support_ratio > 1 ||
    c.max_unsupported_gap < 0 || c.min_continuous_support < 0 ||
    c.min_support_points < 2 || c.max_lines < 1 || c.max_lines > 256)
  {
    throw std::invalid_argument("Invalid shared wall-extractor settings");
  }
  if (mode == "hough") {
    if (c.resolution < 0.01 || c.votes < 1 || c.theta_deg <= 0 || c.hough_max_gap < 0 ||
      c.min_cell_points < 1 || c.min_cell_height < 0 || c.max_grid_side < 8)
    {
      throw std::invalid_argument("Invalid hough wall-extractor settings");
    }
  }
  if (mode == "ransac") {
    if (c.ransac_min_inliers < 2 || c.ransac_max_iterations < 1 ||
      c.ransac_max_iterations > 100000 || c.ransac_downsample_leaf < 0 ||
      c.ransac_distance < 0 ||
      (c.ransac_distance > 1e-9 && c.ransac_distance > c.half_extent))
    {
      throw std::invalid_argument("Invalid RANSAC wall-extractor settings");
    }
  }
  if (mode == "seed_grow") {
    if (c.seed_min_points < 3 || c.seed_contour_angle_deg <= 0 || c.grow_max_gap_bins < 0) {
      throw std::invalid_argument("Invalid seed_grow wall-extractor settings");
    }
  }
}

HoughResult extractHoughWalls(
  const pcl::PointCloud<pcl::PointXYZ> & cloud, const Eigen::Vector2d & center,
  const HoughConfig & c)
{
  validateHoughConfig(c);
  if (!center.allFinite()) {
    throw std::invalid_argument("Wall extractor center is not finite");
  }
  if (isSeedGrowExtractor(c)) {
    return extractSeedGrowWalls(cloud, center, c);
  }
  if (isRansacExtractor(c)) {
    return extractRansacWalls(cloud, center, c);
  }
  return extractHoughWallsGrid(cloud, center, c);
}

namespace
{
WallLine remakeEnds(const WallLine & meta, const Vec & a, const Vec & b)
{
  WallLine out = meta;
  out.a = a;
  out.b = b;
  const double len = (b - a).norm();
  if (len < 1e-9) {return out;}
  const Vec d = (b - a) / len;
  out.normal = Vec(-d.y(), d.x());
  if (out.normal.dot(meta.normal) < 0.0) {out.normal = -out.normal;}
  out.rho = out.normal.dot(a);
  return out;
}
}  // namespace

void snapWallCorners(
  std::vector<WallLine> & lines, double parallel_angle_deg, double snap_dist,
  double min_line_length)
{
  if (lines.size() < 2 || !(snap_dist > 0.0)) {return;}
  const double sin_parallel = std::sin(std::max(1.0, parallel_angle_deg) * pi / 180.0);
  for (size_t i = 0; i < lines.size(); ++i) {
    for (size_t j = i + 1; j < lines.size(); ++j) {
      WallLine & li = lines[i];
      WallLine & lj = lines[j];
      const double len_i = (li.b - li.a).norm();
      const double len_j = (lj.b - lj.a).norm();
      if (len_i < 1e-6 || len_j < 1e-6) {continue;}
      const Vec di = (li.b - li.a) / len_i;
      const Vec dj = (lj.b - lj.a) / len_j;
      const double det = cross(di, dj);
      if (std::abs(det) < sin_parallel) {continue;}
      const Vec offset = lj.a - li.a;
      const double t = cross(offset, dj) / det;
      const double u = cross(offset, di) / det;
      // Intersection must lie near both segments (allow snap_dist beyond ends).
      if (t < -snap_dist || t > len_i + snap_dist ||
        u < -snap_dist || u > len_j + snap_dist)
      {
        continue;
      }
      const Vec p = li.a + t * di;
      const double end_i = std::min(std::abs(t), std::abs(t - len_i));
      const double end_j = std::min(std::abs(u), std::abs(u - len_j));
      if (end_i > snap_dist || end_j > snap_dist) {continue;}

      const bool snap_a_i = (li.a - p).norm() <= (li.b - p).norm();
      const bool snap_a_j = (lj.a - p).norm() <= (lj.b - p).norm();
      Vec ai = snap_a_i ? p : li.a;
      Vec bi = snap_a_i ? li.b : p;
      Vec aj = snap_a_j ? p : lj.a;
      Vec bj = snap_a_j ? lj.b : p;
      if ((bi - ai).norm() < min_line_length || (bj - aj).norm() < min_line_length) {
        continue;
      }
      li = remakeEnds(li, ai, bi);
      lj = remakeEnds(lj, aj, bj);
    }
  }
}

void clampWallsToSupportCloud(
  std::vector<WallLine> & lines, const pcl::PointCloud<pcl::PointXYZ> & cloud,
  double support_band, double margin, double min_line_length)
{
  if (lines.empty() || cloud.empty() || !(support_band > 0.0)) {return;}
  const double m = std::max(0.0, margin);
  std::vector<WallLine> kept;
  kept.reserve(lines.size());
  for (const WallLine & line : lines) {
    const double len = (line.b - line.a).norm();
    if (len < 1e-9) {continue;}
    const Vec d = (line.b - line.a) / len;
    Vec n = line.normal;
    const double nn = n.norm();
    if (nn > 1e-9) {n /= nn;} else {n = Vec(-d.y(), d.x());}
    const double rho = n.dot(line.a);

    double t_min = std::numeric_limits<double>::infinity();
    double t_max = -std::numeric_limits<double>::infinity();
    double z_min = std::numeric_limits<double>::infinity();
    double z_max = -std::numeric_limits<double>::infinity();
    size_t hits = 0;
    for (const auto & p : cloud.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {continue;}
      const Vec xy(p.x, p.y);
      if (std::abs(n.dot(xy) - rho) > support_band) {continue;}
      const double t = d.dot(xy - line.a);
      // Only points that project onto the current segment (no growth past ends).
      if (t < -1e-6 || t > len + 1e-6) {continue;}
      t_min = std::min(t_min, t);
      t_max = std::max(t_max, t);
      z_min = std::min(z_min, static_cast<double>(p.z));
      z_max = std::max(z_max, static_cast<double>(p.z));
      ++hits;
    }
    if (hits < 2 || !std::isfinite(t_min) || !std::isfinite(t_max)) {continue;}
    double lo = std::max(0.0, t_min - m);
    double hi = std::min(len, t_max + m);
    if (hi - lo < min_line_length) {continue;}
    WallLine out = remakeEnds(line, line.a + d * lo, line.a + d * hi);
    if (std::isfinite(z_min) && std::isfinite(z_max) && z_max >= z_min) {
      out.z_min = z_min - m;
      out.z_max = z_max + m;
    }
    out.support_points = hits;
    kept.push_back(out);
  }
  lines.swap(kept);
}

}  // namespace wall_plane_filter
