# wall_plane_filter

3D 激光主路径：

1. 雷达坐标系里，高度在 `[lidar_z_cut_down, lidar_z_cut_up]` 且到雷达 z 轴的水平距离小于 `lidar_z_cut_range_max`（默认 1.5 m）的点删掉，进 `cloud_deviate`。更远的同一高度墙点保留。`base_link` 的 `z_min/z_max` 不变。
2. 剩下的点按每一圈激光看：先去掉方位角上的距离线性趋势。只删比趋势更近、拱高在 `[edge_min, edge_max]`、左右都回到趋势、角宽明显小于 `ring_bow_half_deg`、且拱高不超过 `ap_keep_depth` 的浅弓。斜墙的单调距离变化保留。

旧的侧向分位、射线夹角、窄道空档、体素剔除非结构、MLS 默认关闭。

看 `/wall_plane_filter/cloud_filtered` 和 `cloud_deviate`。

- 鼓包还在 → 加大 `lidar_z_cut_up` / `lidar_z_cut_down` 的幅度，或增大 `edge_max`
- 墙被掏空一条 → 减小高度带
- 误删箱子 → 减小 `edge_max`，或升高 `ap_keep_depth`

```bash
colcon build --packages-select wall_plane_filter --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

深度相机独立发布，不合并四路，也不与雷达点云合并：

- `/wall_plane_filter/camera1/depth_cloud_filtered`（camera2、camera3、camera4 同样命名）
- `/wall_plane_filter/camera1/depth_cloud_deviate`（各路剔除点）

每路保留自己的输入时间戳，坐标系为 `depth_wall_output_frame`。原 `cloud_filtered/cloud_deviate` 仅发布雷达数据。找墙仍在内部融合四路证据。

深度过滤点云在各相机回调完成后立即发布；雷达过滤点云在雷达回调处理完后立即发布。两者都不合并，也不等找墙。10 Hz 的融合周期只汇集深度找墙证据、发深度墙 Marker，并按最新墙线再刷一次深度点云的墙带剔除。

跟踪后的竖直墙面几何：

- `/wall_plane_filter/depth_wall_planes`（`wall_plane_filter/WallPlaneArray`）
- `/wall_plane_filter/lidar_wall_planes`

每条墙：`normal·[x,y]=rho`，高度 `[z_min,z_max]`，水平跨度 `a→b`，并带四角 `corners`。同话题 Marker 里还有半透明 `TRIANGLE_LIST` 面片。

## WSL bag + 容器节点 + WSL RViz

三边必须用同一套 Cyclone 配置，否则话题能发现、点云仍是空的：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///workspace/dds_local.xml
export ROS_LOCALHOST_ONLY=0
```

不要用 `/capella/dds.xml`：它绑 `docker0`，本机 `docker0` 是 down。`dds_local.xml` 强制 `eth0`、DDS 分片 1400B，并把 Best-Effort 重组槽从默认 4 提到 64（RViz 同时订 4 路深度+雷达时，默认 4 会直接丢点云）。

过滤点云同时发 Reliable 和 Best Effort。你这边 RViz 点云是 Reliable，必须能对上 Reliable 发布端。Fixed Frame 用 `map`。回放要 `ros2 bag play --clock`。
# wall_plane_filter
