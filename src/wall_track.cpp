#include "wall_plane_filter/wall_track.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace wall_plane_filter
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kSlop = 0.02;

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

bool associated(
  const WallLine & track, const WallLine & obs, const WallTrackConfig & c, double & score,
  double normal_gate)
{
  const double turn = std::abs(wrapPi(directionAngle(track) - directionAngle(obs)));
  const double parallel = std::min(turn, kPi - turn);
  if (parallel > c.merge_angle_deg * kPi / 180.0) {return false;}
  const double d0 = std::abs(track.normal.dot(obs.a) - track.rho);
  const double d1 = std::abs(track.normal.dot(obs.b) - track.rho);
  const double normal = std::max(d0, d1);
  if (normal > normal_gate) {return false;}
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

double associationGate(const WallTracker::Track & track, double stamp_sec, const WallTrackConfig & c)
{
  double gate = c.normal_gate;
  if (!track.confirmed || !(c.coast_gate_extra > 0.0) || !(c.lose_sec > 1e-6)) {
    return gate;
  }
  const double age = stamp_sec - track.last_seen;
  if (age <= 0.0) {
    return gate;
  }
  const double scale = std::min(1.0, age / c.lose_sec);
  gate += c.coast_gate_extra * scale;
  // Optional: widen a bit with KF rho uncertainty while coasting.
  if (track.kf_valid && track.P(0, 0) > 0.0) {
    gate += std::min(0.10, 2.0 * std::sqrt(track.P(0, 0)));
  }
  return gate;
}

WallTracker::VecX lineToState(const WallLine & line)
{
  WallTracker::VecX x = WallTracker::VecX::Zero();
  Eigen::Vector2d n = line.normal;
  double rho = line.rho;
  const double nrm = n.norm();
  if (nrm > 1e-9) {
    n /= nrm;
  } else {
    n = Eigen::Vector2d(0.0, 1.0);
  }
  // Prefer normal consistent with endpoints (left-handed to direction a→b).
  Eigen::Vector2d dir = (line.b - line.a);
  if (dir.norm() > 1e-9) {
    dir.normalize();
    const Eigen::Vector2d n_from_dir(-dir.y(), dir.x());
    if (n.dot(n_from_dir) < 0.0) {
      n = -n;
      rho = -rho;
    }
    dir = Eigen::Vector2d(-n.y(), n.x());
  } else {
    dir = Eigen::Vector2d(-n.y(), n.x());
  }
  const Eigen::Vector2d origin = rho * n;
  double s0 = dir.dot(line.a - origin);
  double s1 = dir.dot(line.b - origin);
  if (s0 > s1) {
    std::swap(s0, s1);
  }
  x(0) = rho;
  x(1) = std::atan2(n.y(), n.x());
  x(2) = s0;
  x(3) = s1;
  return x;
}

WallLine stateToLine(const WallTracker::VecX & x, const WallLine & meta)
{
  const Eigen::Vector2d normal(std::cos(x(1)), std::sin(x(1)));
  const Eigen::Vector2d dir(-normal.y(), normal.x());
  const Eigen::Vector2d origin = x(0) * normal;
  double lo = x(2);
  double hi = x(3);
  if (lo > hi) {
    std::swap(lo, hi);
  }
  return makeSegment(origin + lo * dir, origin + hi * dir, meta);
}

void initKf(WallTracker::Track & track, const WallTrackConfig & c)
{
  track.x = lineToState(track.line);
  track.P = WallTracker::MatP::Zero();
  track.P(0, 0) = std::max(1e-4, c.kf_r_rho * 4.0);
  track.P(1, 1) = std::max(1e-4, c.kf_r_theta * 4.0);
  track.P(2, 2) = std::max(1e-3, c.kf_r_end * 4.0);
  track.P(3, 3) = std::max(1e-3, c.kf_r_end * 4.0);
  track.kf_valid = true;
}

void predictKf(WallTracker::Track & track, double dt, const WallTrackConfig & c)
{
  if (!track.kf_valid) {
    initKf(track, c);
    return;
  }
  // Static wall in map: mean unchanged; covariance grows with time.
  const double t = std::clamp(dt, 0.0, std::max(0.05, c.lose_sec));
  const double scale = std::max(t, 1e-3);
  track.P(0, 0) += c.kf_q_rho * scale;
  track.P(1, 1) += c.kf_q_theta * scale;
  track.P(2, 2) += c.kf_q_end * scale;
  track.P(3, 3) += c.kf_q_end * scale;
}

void syncLineFromState(WallTracker::Track & track, const WallLine & meta)
{
  Eigen::Vector2d prefer_dir = track.line.b - track.line.a;
  track.line = stateToLine(track.x, meta);
  if (prefer_dir.squaredNorm() > 1e-12 &&
    (track.line.b - track.line.a).dot(prefer_dir) < 0.0)
  {
    std::swap(track.line.a, track.line.b);
    track.line.normal = -track.line.normal;
    track.line.rho = -track.line.rho;
    // Keep KF theta consistent with stored normal.
    track.x(0) = track.line.rho;
    track.x(1) = std::atan2(track.line.normal.y(), track.line.normal.x());
    const Eigen::Vector2d origin = track.x(0) * track.line.normal;
    const Eigen::Vector2d dir(-track.line.normal.y(), track.line.normal.x());
    double s0 = dir.dot(track.line.a - origin);
    double s1 = dir.dot(track.line.b - origin);
    if (s0 > s1) {
      std::swap(s0, s1);
    }
    track.x(2) = s0;
    track.x(3) = s1;
  }
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
  double lo = old_lo;
  double hi = old_hi;
  if (c.clamp_extent_to_obs) {
    // Never longer than current observation (support-derived); shrink immediately.
    lo = new_lo;
    hi = new_hi;
    track.shorten_lo = 0;
    track.shorten_hi = 0;
  } else if (new_lo < old_lo - kSlop) {
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
  if (!c.clamp_extent_to_obs) {
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
  }
  if (hi - lo < c.min_line_length) {
    track.confirmed = false;
    track.hits = 0;
    track.kf_valid = false;
    return;
  }
  WallLine meta = obs;
  if (std::isfinite(track.line.z_min) && std::isfinite(obs.z_min)) {
    meta.z_min = (1.0 - w) * track.line.z_min + w * obs.z_min;
  }
  if (std::isfinite(track.line.z_max) && std::isfinite(obs.z_max)) {
    meta.z_max = (1.0 - w) * track.line.z_max + w * obs.z_max;
  }
  track.line = makeSegment(origin + lo * dir, origin + hi * dir, meta);
  if (track.line.normal.dot(normal) < 0.0) {
    track.line.normal = -track.line.normal;
    track.line.rho = -track.line.rho;
    std::swap(track.line.a, track.line.b);
  }
}

void kfUpdate(WallTracker::Track & track, const WallLine & obs, double stamp_sec, const WallTrackConfig & c)
{
  if (!track.kf_valid) {
    initKf(track, c);
  }
  const double dt = stamp_sec - track.last_seen;
  predictKf(track, dt, c);

  // Align observation normal to track.
  WallLine aligned = obs;
  if (aligned.normal.dot(track.line.normal) < 0.0) {
    aligned.normal = -aligned.normal;
    aligned.rho = -aligned.rho;
    std::swap(aligned.a, aligned.b);
  }
  const WallTracker::VecX z_raw = lineToState(aligned);
  WallTracker::VecX z = z_raw;
  // Theta relative to current state (unwrap).
  z(1) = track.x(1) + wrapPi(z_raw(1) - track.x(1));

  // Endpoint policy: by default grow immediately / shorten gated.
  // With clamp_extent_to_obs: trust observation ends (no grow past support).
  WallTracker::MatP R = WallTracker::MatP::Zero();
  const double sigma = std::isfinite(obs.sigma) ? std::max(0.0, obs.sigma) : 0.0;
  R(0, 0) = std::max(c.kf_r_rho, sigma * sigma);
  R(1, 1) = std::max(1e-6, c.kf_r_theta);
  R(2, 2) = std::max(1e-6, c.kf_r_end);
  R(3, 3) = std::max(1e-6, c.kf_r_end);

  if (c.clamp_extent_to_obs) {
    track.shorten_lo = 0;
    track.shorten_hi = 0;
    R(2, 2) = std::max(1e-6, c.kf_r_end * 0.25);
    R(3, 3) = std::max(1e-6, c.kf_r_end * 0.25);
  } else if (z(2) < track.x(2) - kSlop) {
    // Grow lo.
    track.shorten_lo = 0;
    R(2, 2) = std::max(1e-6, c.kf_r_end * 0.25);
  } else if (z(2) > track.x(2) + kSlop) {
    ++track.shorten_lo;
    if (track.shorten_lo < c.shorten_frames) {
      z(2) = track.x(2);
      R(2, 2) = 1e6;
    } else {
      track.shorten_lo = 0;
      R(2, 2) = std::max(1e-6, c.kf_r_end);
    }
  } else {
    track.shorten_lo = 0;
  }

  if (!c.clamp_extent_to_obs) {
    if (z(3) > track.x(3) + kSlop) {
      // Grow hi.
      track.shorten_hi = 0;
      R(3, 3) = std::max(1e-6, c.kf_r_end * 0.25);
    } else if (z(3) < track.x(3) - kSlop) {
      ++track.shorten_hi;
      if (track.shorten_hi < c.shorten_frames) {
        z(3) = track.x(3);
        R(3, 3) = 1e6;
      } else {
        track.shorten_hi = 0;
        R(3, 3) = std::max(1e-6, c.kf_r_end);
      }
    } else {
      track.shorten_hi = 0;
    }
  }

  // Linear KF, H = I.
  const WallTracker::MatP S = track.P + R;
  // Diagonal-dominant inverse (states weakly coupled).
  WallTracker::MatP Sinv = WallTracker::MatP::Zero();
  for (int i = 0; i < WallTracker::kDim; ++i) {
    Sinv(i, i) = 1.0 / std::max(1e-12, S(i, i));
  }
  const WallTracker::MatP K = track.P * Sinv;
  WallTracker::VecX innov = z - track.x;
  innov(1) = wrapPi(innov(1));
  track.x += K * innov;
  track.x(1) = wrapPi(track.x(1));
  if (track.x(2) > track.x(3)) {
    std::swap(track.x(2), track.x(3));
    std::swap(track.P(2, 2), track.P(3, 3));
  }
  const WallTracker::MatP I = WallTracker::MatP::Identity();
  track.P = (I - K) * track.P;
  // Keep P SPD-ish on diagonal.
  for (int i = 0; i < WallTracker::kDim; ++i) {
    track.P(i, i) = std::max(1e-12, track.P(i, i));
    for (int j = 0; j < WallTracker::kDim; ++j) {
      if (i != j) {
        track.P(i, j) = 0.0;
      }
    }
  }

  if (track.x(3) - track.x(2) < c.min_line_length) {
    track.confirmed = false;
    track.hits = 0;
    track.kf_valid = false;
    return;
  }

  const double w = std::clamp(c.obs_weight, 0.0, 1.0);
  WallLine meta = obs;
  if (std::isfinite(track.line.z_min) && std::isfinite(obs.z_min)) {
    meta.z_min = (1.0 - w) * track.line.z_min + w * obs.z_min;
  }
  if (std::isfinite(track.line.z_max) && std::isfinite(obs.z_max)) {
    meta.z_max = (1.0 - w) * track.line.z_max + w * obs.z_max;
  }
  meta.sigma = obs.sigma;
  meta.support_ratio = obs.support_ratio;
  meta.support_points = obs.support_points;
  syncLineFromState(track, meta);
}

void fuseObservation(
  WallTracker::Track & track, const WallLine & obs, double stamp_sec, const WallTrackConfig & c)
{
  if (c.kf_enable) {
    kfUpdate(track, obs, stamp_sec, c);
  } else {
    blend(track, obs, c);
    if (track.hits > 0) {
      initKf(track, c);
    }
  }
}
}  // namespace

WallTrackConfig trackConfigFrom(const HoughConfig & hough, const WallTrackConfig & base)
{
  WallTrackConfig config = base;
  config.merge_angle_deg = hough.merge_angle_deg;
  // Prefer the larger gap so association behaves like a along-track buffer.
  config.merge_gap = std::max(base.merge_gap, hough.merge_gap);
  config.min_line_length = hough.min_line_length;
  return config;
}

void WallTracker::clear()
{
  tracks_.clear();
  next_id_ = 1;
}

std::vector<WallLine> WallTracker::update(
  const std::vector<WallLine> & observations, double stamp_sec, const WallTrackConfig & config,
  WallTrackUpdateStats * stats)
{
  WallTrackUpdateStats local;
  WallTrackUpdateStats & st = stats ? *stats : local;
  st = WallTrackUpdateStats{};
  st.obs = static_cast<int>(observations.size());
  st.tracks_in = static_cast<int>(tracks_.size());

  const int n_obs = st.obs;
  const int n_trk = st.tracks_in;
  struct Pair
  {
    int track;
    int obs;
    double score;
  };
  std::vector<Pair> pairs;
  for (int t = 0; t < n_trk; ++t) {
    const double gate = associationGate(tracks_[static_cast<size_t>(t)], stamp_sec, config);
    for (int o = 0; o < n_obs; ++o) {
      double score = 0.0;
      if (associated(
          tracks_[static_cast<size_t>(t)].line, observations[static_cast<size_t>(o)], config, score,
          gate))
      {
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
      fuseObservation(track, observations[static_cast<size_t>(obs)], stamp_sec, config);
      if (track.hits <= 0) {continue;}
      track.hits = std::min(track.hits + 1, config.confirm_frames);
      track.last_seen = stamp_sec;
      track.confirmed = track.hits >= config.confirm_frames;
      ++st.matched;
      next.push_back(track);
      continue;
    }
    // Unconfirmed tracks die immediately without a match. Confirmed tracks
    // coast: predict covariance only, keep map-frame mean frozen.
    if (!track.confirmed) {
      ++st.dropped_unconfirmed;
      continue;
    }
    const double age = stamp_sec - track.last_seen;
    if (age > config.lose_sec) {
      ++st.expired;
      continue;
    }
    ++st.coasting;
    track.shorten_lo = 0;
    track.shorten_hi = 0;
    if (config.kf_enable) {
      predictKf(track, stamp_sec - track.last_seen, config);
    }
    next.push_back(track);
  }
  for (int o = 0; o < n_obs; ++o) {
    if (o < static_cast<int>(used.size()) && used[static_cast<size_t>(o)]) {continue;}
    const WallLine & line = observations[static_cast<size_t>(o)];
    if ((line.b - line.a).norm() < config.min_line_length) {continue;}
    Track track;
    track.id = next_id_++;
    track.line = line;
    track.last_seen = stamp_sec;
    track.hits = 1;
    track.confirmed = config.confirm_frames <= 1;
    initKf(track, config);
    ++st.spawned;
    next.push_back(track);
  }
  tracks_.swap(next);
  const auto out = confirmed();
  st.confirmed_out = static_cast<int>(out.size());
  return out;
}

std::vector<WallLine> WallTracker::confirmed() const
{
  std::vector<WallLine> published;
  for (const auto & track : tracks_) {
    if (track.confirmed) {published.push_back(track.line);}
  }
  return published;
}


}  // namespace wall_plane_filter
