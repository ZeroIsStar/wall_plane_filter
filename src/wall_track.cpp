#include "wall_plane_filter/wall_track.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace wall_plane_filter
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

double wrapPi(double a)
{
  while (a > kPi) {a -= 2.0 * kPi;}
  while (a < -kPi) {a += 2.0 * kPi;}
  return a;
}

double directionAngle(const WallLine & line)
{
  return std::atan2(line.b.y() - line.a.y(), line.b.x() - line.a.x());
}

WallLine makeSegment(const Eigen::Vector2d & a, const Eigen::Vector2d & b, const WallLine & meta)
{
  WallLine out = meta;
  out.a = a;
  out.b = b;
  const Eigen::Vector2d d = (b - a).normalized();
  out.normal = Eigen::Vector2d(-d.y(), d.x());
  out.rho = out.normal.dot(a);
  return out;
}

bool associated(const WallLine & track, const WallLine & obs, const WallTrackConfig & c, double & score)
{
  const double turn = std::abs(wrapPi(directionAngle(track) - directionAngle(obs)));
  const double parallel = std::min(turn, kPi - turn);
  if (parallel > c.merge_angle_deg * kPi / 180.0) {return false;}
  const double d0 = std::abs(track.normal.dot(obs.a) - track.rho);
  const double d1 = std::abs(track.normal.dot(obs.b) - track.rho);
  const double normal = std::max(d0, d1);
  if (normal > c.normal_gate) {return false;}
  const Eigen::Vector2d dir = (track.b - track.a).normalized();
  const double s0 = 0.0;
  const double s1 = (track.b - track.a).norm();
  const double t0 = dir.dot(obs.a - track.a);
  const double t1 = dir.dot(obs.b - track.a);
  const double lo = std::max(s0, std::min(t0, t1));
  const double hi = std::min(s1, std::max(t0, t1));
  if (lo > hi + c.merge_gap) {return false;}
  score = normal + 0.05 * parallel;
  return true;
}

void blend(WallTracker::Track & track, const WallLine & obs, const WallTrackConfig & c)
{
  Eigen::Vector2d n = obs.normal;
  double rho = obs.rho;
  if (n.dot(track.line.normal) < 0.0) {
    n = -n;
    rho = -rho;
  }
  const double w = std::clamp(c.obs_weight, 0.0, 1.0);
  const double ang = std::atan2(track.line.normal.y(), track.line.normal.x()) +
    w * wrapPi(std::atan2(n.y(), n.x()) - std::atan2(track.line.normal.y(), track.line.normal.x()));
  const Eigen::Vector2d normal(std::cos(ang), std::sin(ang));
  rho = (1.0 - w) * track.line.rho + w * rho;
  Eigen::Vector2d dir(-normal.y(), normal.x());
  if (dir.dot(track.line.b - track.line.a) < 0.0) {dir = -dir;}
  const Eigen::Vector2d origin = rho * normal;
  const auto along = [&](const Eigen::Vector2d & p) {return dir.dot(p - origin);};
  double old_lo = along(track.line.a);
  double old_hi = along(track.line.b);
  if (old_lo > old_hi) {std::swap(old_lo, old_hi);}
  double new_lo = along(obs.a);
  double new_hi = along(obs.b);
  if (new_lo > new_hi) {std::swap(new_lo, new_hi);}
  constexpr double kSlop = 0.02;
  double lo = old_lo;
  double hi = old_hi;
  if (new_lo < old_lo - kSlop) {
    lo = new_lo;
    track.shorten_lo = 0;
  } else if (new_lo > old_lo + kSlop) {
    ++track.shorten_lo;
    if (track.shorten_lo >= c.shorten_frames) {
      lo = new_lo;
      track.shorten_lo = 0;
    }
  } else {
    track.shorten_lo = 0;
  }
  if (new_hi > old_hi + kSlop) {
    hi = new_hi;
    track.shorten_hi = 0;
  } else if (new_hi < old_hi - kSlop) {
    ++track.shorten_hi;
    if (track.shorten_hi >= c.shorten_frames) {
      hi = new_hi;
      track.shorten_hi = 0;
    }
  } else {
    track.shorten_hi = 0;
  }
  if (hi - lo < c.min_line_length) {
    track.confirmed = false;
    track.hits = 0;
    return;
  }
  WallLine meta = obs;
  // Keep a smoothed height band for the vertical wall plane model.
  if (std::isfinite(track.line.z_min) && std::isfinite(obs.z_min)) {
    meta.z_min = (1.0 - w) * track.line.z_min + w * obs.z_min;
  }
  if (std::isfinite(track.line.z_max) && std::isfinite(obs.z_max)) {
    meta.z_max = (1.0 - w) * track.line.z_max + w * obs.z_max;
  }
  track.line = makeSegment(origin + lo * dir, origin + hi * dir, meta);
  // Keep the smoothed normal if endpoint rebuild flipped it.
  if (track.line.normal.dot(normal) < 0.0) {
    track.line.normal = -track.line.normal;
    track.line.rho = -track.line.rho;
    std::swap(track.line.a, track.line.b);
  }
}
}  // namespace

WallTrackConfig trackConfigFrom(const HoughConfig & hough)
{
  WallTrackConfig config;
  config.merge_angle_deg = hough.merge_angle_deg;
  config.merge_gap = hough.merge_gap;
  config.min_line_length = hough.min_line_length;
  return config;
}

void WallTracker::clear() {tracks_.clear();}

std::vector<WallLine> WallTracker::update(
  const std::vector<WallLine> & observations, double stamp_sec, const WallTrackConfig & config)
{
  const int n_obs = static_cast<int>(observations.size());
  const int n_trk = static_cast<int>(tracks_.size());
  struct Pair
  {
    int track;
    int obs;
    double score;
  };
  std::vector<Pair> pairs;
  for (int t = 0; t < n_trk; ++t) {
    for (int o = 0; o < n_obs; ++o) {
      double score = 0.0;
      if (associated(tracks_[static_cast<size_t>(t)].line, observations[static_cast<size_t>(o)], config, score)) {
        pairs.push_back(Pair{t, o, score});
      }
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const Pair & a, const Pair & b) {return a.score < b.score;});
  std::vector<int> match(static_cast<size_t>(n_trk), -1);
  std::vector<char> used(static_cast<size_t>(std::max(0, n_obs)), 0);
  for (const auto & pair : pairs) {
    if (match[static_cast<size_t>(pair.track)] >= 0 || used[static_cast<size_t>(pair.obs)]) {continue;}
    match[static_cast<size_t>(pair.track)] = pair.obs;
    used[static_cast<size_t>(pair.obs)] = 1;
  }

  std::vector<Track> next;
  next.reserve(tracks_.size() + observations.size());
  for (int t = 0; t < n_trk; ++t) {
    Track track = tracks_[static_cast<size_t>(t)];
    const int obs = match[static_cast<size_t>(t)];
    if (obs >= 0) {
      blend(track, observations[static_cast<size_t>(obs)], config);
      if (track.hits <= 0) {continue;}
      track.hits = std::min(track.hits + 1, config.confirm_frames);
      track.last_seen = stamp_sec;
      track.confirmed = track.hits >= config.confirm_frames;
      next.push_back(track);
      continue;
    }
    // An unconfirmed streak has no support this frame. A published line stays
    // until it has been missing longer than lose_sec.
    if (!track.confirmed) {continue;}
    if (stamp_sec - track.last_seen > config.lose_sec) {continue;}
    next.push_back(track);
  }
  for (int o = 0; o < n_obs; ++o) {
    if (o < static_cast<int>(used.size()) && used[static_cast<size_t>(o)]) {continue;}
    const WallLine & line = observations[static_cast<size_t>(o)];
    if ((line.b - line.a).norm() < config.min_line_length) {continue;}
    Track track;
    track.line = line;
    track.last_seen = stamp_sec;
    track.hits = 1;
    track.confirmed = config.confirm_frames <= 1;
    next.push_back(track);
  }
  tracks_.swap(next);
  std::vector<WallLine> published;
  for (const auto & track : tracks_) {
    if (track.confirmed) {published.push_back(track.line);}
  }
  return published;
}

}  // namespace wall_plane_filter
