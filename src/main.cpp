#include "wall_plane_filter/wall_plane_filter_node.hpp"

#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  rclcpp::spin(std::make_shared<wall_plane_filter::WallPlaneFilterNode>(options));
  rclcpp::shutdown();
  return 0;
}
