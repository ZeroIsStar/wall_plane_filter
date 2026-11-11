#pragma once

#include "wall_plane_filter/wall_hough.hpp"

#include <Eigen/Core>
#include <vector>

namespace wall_plane_filter
{

struct WallTrackConfig
{
  double merge_angle_deg{5.0};
  double normal_gate{0.15};
  double merge_gap{0.25};
  double obs_weight{0.25};
  int confirm_frames{3};
  int shorten_frames{5};
  double lose_sec{1.50};
  double coast_gate_extra{0.15};
  double min_line_length{0.50};
  /** If true, matched updates take observation endpoints (no grow past support/obs). */
  bool clamp_extent_to_obs{true};
  bool kf_enable{true};
  double kf_q_rho{1.0e-4};
  double kf_q_theta{1.0e-4};
  double kf_q_end{2.0e-2};
  double kf_r_rho{2.5e-3};
  double kf_r_theta{1.0e-3};
  double kf_r_end{1.0e-2};
};

struct WallTrackUpdateStats
{
  int obs{0};
  int tracks_in{0};
  int matched{0};
  int spawned{0};
  int coasting{0};
  int expired{0};
  int dropped_unconfirmed{0};
  int confirmed_out{0};
};

class WallTracker
{
public:
  static constexpr int kDim = 4;
  using VecX = Eigen::Matrix<double, kDim, 1>;
  using MatP = Eigen::Matrix<double, kDim, kDim>;

  struct Track
  {
    int id{0};
    WallLine line;
    double last_seen{0};
    int hits{0};
    int shorten_lo{0};
    int shorten_hi{0};
    bool confirmed{false};
    bool kf_valid{false};
    VecX x{VecX::Zero()};
    MatP P{MatP::Identity()};
  };

  void clear();
  std::vector<WallLine> update(
    const std::vector<WallLine> & observations, double stamp_sec, const WallTrackConfig & config,
    WallTrackUpdateStats * stats = nullptr);
  std::vector<WallLine> confirmed() const;

private:
  std::vector<Track> tracks_;
  int next_id_{1};
};

WallTrackConfig trackConfigFrom(const HoughConfig & hough, const WallTrackConfig & base = {});

}  // namespace wall_plane_filter
