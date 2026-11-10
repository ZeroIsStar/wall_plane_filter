#pragma once

#include "wall_plane_filter/wall_hough.hpp"

#include <vector>

namespace wall_plane_filter
{

// Temporal association of wall segments already expressed in the map frame.
// Rho/theta are blended here. Do not feed base_link lines: ego-motion would
// be smoothed into the wall.
struct WallTrackConfig
{
  double merge_angle_deg{5.0};
  // Association gate on |n·endpoint - rho| (m). Acts like a normal buffer.
  double normal_gate{0.15};
  // Along-track gap allowed when matching segments (m).
  double merge_gap{0.25};
  // Weight of the new observation. Lower = smoother, less flicker.
  double obs_weight{0.25};
  int confirm_frames{3};
  // Frames of shorter observation before endpoints retract (grow is immediate).
  int shorten_frames{5};
  // Keep publishing a confirmed wall this long after last match (s).
  double lose_sec{1.50};
  double min_line_length{0.50};
};

class WallTracker
{
public:
  struct Track
  {
    WallLine line;
    double last_seen{0};
    int hits{0};
    int shorten_lo{0};
    int shorten_hi{0};
    bool confirmed{false};
  };

  void clear();
  // `stamp_sec` is the source cloud stamp. Only confirmed lines are returned.
  std::vector<WallLine> update(
    const std::vector<WallLine> & observations, double stamp_sec, const WallTrackConfig & config);
  std::vector<WallLine> confirmed() const;

private:
  std::vector<Track> tracks_;
};

// Fill association geometry from Hough; temporal fields come from `base`.
WallTrackConfig trackConfigFrom(const HoughConfig & hough, const WallTrackConfig & base = {});

}  // namespace wall_plane_filter
