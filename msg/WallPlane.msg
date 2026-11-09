# Finite vertical wall plane in the parent WallPlaneArray header frame.
# Geometry: normal.x * X + normal.y * Y = rho,  Z in [z_min, z_max],
# horizontal span from endpoint a to endpoint b (XY). normal.z is always 0.
geometry_msgs/Vector3 normal
float64 rho
geometry_msgs/Point a
geometry_msgs/Point b
float64 z_min
float64 z_max
# Convenience rectangle corners in order: a@z_min, b@z_min, b@z_max, a@z_max.
geometry_msgs/Point[4] corners
float64 length
float64 height
float64 sigma
float64 support_ratio
uint32 support_points
# 0 = lidar, 1 = depth
uint8 source
uint8 SOURCE_LIDAR=0
uint8 SOURCE_DEPTH=1
