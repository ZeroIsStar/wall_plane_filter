std_msgs/Header header
# Door state in front of the robot (base_link / target_frame).
uint8 UNKNOWN=0
uint8 OPEN=1
uint8 CLOSED=2
uint8 state
# Estimated clear gap across the door width (m). Larger → more open.
float64 gap_m
# Median range of "near" columns when closed-like (m); NaN if unknown.
float64 door_range_m
# Fraction of lateral bins classified as open gap [0,1].
float64 open_ratio
float64 confidence
uint32 valid_bins
uint32 open_bins
uint32 near_bins
