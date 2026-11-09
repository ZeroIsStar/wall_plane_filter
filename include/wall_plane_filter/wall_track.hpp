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
  double normal_gate{0.12};
  double merge_gap{0.15};
  // Weight of the new observation. The previous map line keeps the rest.
  double obs_weight{0.35};
  int confirm_frames{3};
  int shorten_frames{3};
  double lose_sec{0.30};
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

private:
  std::vector<Track> tracks_;
};

WallTrackConfig trackConfigFrom(const HoughConfig & hough);

}  // namespace wall_plane_filter
