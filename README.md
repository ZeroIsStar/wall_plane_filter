# wall_plane_filter

从深度点云与 3D 雷达提取竖直墙段：预处理 → 找线 `extractor`（`hough` | `ransac` | `seed_grow`）→ `WallTracker`+KF → 支撑夹紧 / 拐角吸附 → 发布墙平面、Marker 与过滤点云。

默认雷达与深度 ROI / 找墙水平距离均为 **3 m**。墙面几何夹紧到本帧支撑点云，不长出贴墙范围；L 形拐角将靠近交点的端点吸附对齐。

- `*_filtered` = 过滤后**保留**点  
- `*_deviate` = 剔除点（ROR / SOR / 墙旁噪声带）

---

## 构建与启动

```bash
# 工作空间根目录
colcon build --packages-select wall_plane_filter
source install/setup.bash

# 推荐：launch（加载 share/.../config/wall_plane_filter.yaml）
ros2 launch wall_plane_filter wall_plane_filter.launch.py

# 指定参数文件
ros2 launch wall_plane_filter wall_plane_filter.launch.py \
  params_file:=/src/wall_plane_filter/config/wall_plane_filter.yaml

# 或直接跑节点
ros2 run wall_plane_filter wall_plane_filter_node --ros-args \
  --params-file /src/wall_plane_filter/config/wall_plane_filter.yaml
```

依赖：深度/雷达点云话题、以及 `target_frame`（默认 `base_link`）与 `depth_wall_map_frame`（默认 `map`）之间的 TF。

---

## 处理流程

### 深度（每路相机）

1. TF 到 `target_frame`，ROI 裁剪（高度 + 水平距离）  
2. 各向同性 `VoxelGrid`（`depth_voxel_xy`³）  
3. 点数 ≥ `depth_ror_min_points` 时 **ROR 粗剔** → **SOR 精修**（剔除进 deviate）  
4. 几何 Bilateral 平滑 → `depth_cloud_filtered`  
5. TF 到 `map`，按 `depth_hough.extractor` 找线 → 每相机 `WallTracker`+KF  
6. `clampWallsToSupportCloud` + `snapWallCorners`  
7. 多相机融合墙线（用于发布前 `cullWallBand`）  
8. 发布 `wall_planes` / `wall_markers`；对点云做墙带投影/剔除后发布 filtered / deviate  

### 雷达

1. TF 到 `target_frame`，ROI + 各向同性体素  
2. 可选发布 `/wall_plane_filter/cloud_filtered`  
3. 水平距离 ≤ `lidar_wall_range_max` 找墙 → `lidar_hough.extractor` → 共用 `wall_track` 的 `lidar_tracks_`  
4. 支撑夹紧 + 拐角吸附 → `/wall_plane_filter/lidar_wall_planes`、`lidar_wall_markers`  

### extractor

| 名称 | 含义 |
|------|------|
| `hough` | XY 占据栅格 + OpenCV `HoughLinesP`（当前 yaml 默认） |
| `ransac` | XY 多模型直线迭代 |
| `seed_grow` | 极坐标最近轮廓 → 种子段 → 区域生长 |

`depth_hough` 与 `lidar_hough` 可分别配置。

---

## 话题

### 订阅

| 话题 | 类型 | 说明 |
|------|------|------|
| `cloud_topic`（默认 `/vanjee/lidar`） | `sensor_msgs/PointCloud2` | 雷达 |
| `depth_topics[]`（如 `/cameraN/depth/points`） | `sensor_msgs/PointCloud2` | 深度 |

### 发布

| 话题 | 说明 |
|------|------|
| `/wall_plane_filter/<cam>/depth_cloud_filtered` | 深度过滤后保留点 |
| `/wall_plane_filter/<cam>/depth_cloud_deviate` | 深度剔除点 |
| `/wall_plane_filter/<cam>/wall_planes` | 该相机墙平面数组 |
| `/wall_plane_filter/<cam>/wall_markers` | 该相机墙 Marker |
| `/wall_plane_filter/lidar_wall_planes` | 雷达墙平面 |
| `/wall_plane_filter/lidar_wall_markers` | 雷达墙 Marker |
| `/wall_plane_filter/cloud_filtered` | 可选，雷达 ROI+体素后点云 |

`WallPlane`：竖直墙矩形（法向在 XY，`a`/`b` 水平端点，`z_min`/`z_max`，`corners[4]`）。

---

## 参数说明

配置文件：`config/wall_plane_filter.yaml`（节点名 `wall_plane_filter`）。以下与现逻辑一致；未列单位的布尔/整型见注释。

### 通用 / TF / 显示

| 参数 | 含义 |
|------|------|
| `target_frame` | ROI 裁剪坐标系（默认 `base_link`） |
| `tf_timeout` | TF 查询超时（秒） |
| `wall_line_width` | Marker 线宽（米） |
| `wall_display_z` | 显示参考高度（米） |
| `wall_marker_lifetime` | Marker 生命周期（秒） |
| `wall_marker_publish_hz` | Marker 发布上限 Hz；≤0 每帧都发 |

### 雷达

| 参数 | 含义 |
|------|------|
| `cloud_topic` | 雷达点云话题 |
| `lidar_wall_enable` | 是否找墙并发布雷达墙 |
| `publish_cloud_filtered` | 是否发布雷达 `cloud_filtered` |
| `z_min` / `z_max` | 雷达 ROI 高度带（米） |
| `range_min` / `range_max` | 雷达 ROI 水平距离（米） |
| `voxel_leaf` | 雷达各向同性体素边长（米） |
| `lidar_wall_range_max` | 找墙水平距离上限（米，≤ `range_max`） |

### 拐角 / 支撑夹紧（深度与雷达共用）

| 参数 | 含义 |
|------|------|
| `corner_snap_enable` | 是否将 L 形墙端点吸附到交点 |
| `corner_snap_dist` | 吸附距离（米）；≤0 用对应 `intersection_extension` |
| `wall_clamp_to_support` | 是否按支撑点云硬夹紧墙段 a–b / z |
| `wall_extent_margin_m` | 夹紧相对支撑极值的余量（米） |

### 深度相机与预处理

| 参数 | 含义 |
|------|------|
| `depth_clean_enable` | 是否启用深度预处理与发布 |
| `depth_topics` | 深度点云话题列表 |
| `depth_wall_cameras` | 参与找墙的相机名（与话题路径解析出的 cam 名对应） |
| `depth_wall_map_frame` | 跟踪/融合工作系（默认 `map`） |
| `depth_wall_output_frame` | 发布系（默认 `map`） |
| `depth_z_min` / `depth_z_max` | 深度 ROI 高度（米） |
| `depth_range_min` / `depth_range_max` | 深度 ROI 水平距离（米） |
| `depth_voxel_xy` | 各向同性体素边长 leaf（米），即 leaf×leaf×leaf |
| `depth_ror_enable` | 是否启用 RadiusOutlierRemoval 粗剔 |
| `depth_ror_min_points` | 体素后点数少于此则跳过 ROR |
| `depth_ror_radius` | ROR 搜索半径（米） |
| `depth_ror_min_neighbors` | 半径内最少邻域点数 |
| `depth_sor_enable` | 是否启用 StatisticalOutlierRemoval 精修 |
| `depth_sor_mean_k` | SOR 邻域点数 K |
| `depth_sor_stddev_mul` | SOR 标准差倍数（越大越宽松） |
| `depth_bilateral_enable` | 是否几何双边平滑 |
| `depth_bilateral_sigma_s` | 双边空间 sigma（米） |
| `depth_bilateral_sigma_r` | 双边值域 sigma（\|Δz\|，米） |

### 发布前墙带裁剪

| 参数 | 含义 |
|------|------|
| `depth_wall_band_m` | 贴墙内带：保留，可投影到墙（米） |
| `depth_wall_band_outer_m` | 外带上限；介于 band~outer 进 deviate（米） |
| `depth_wall_project` | 内带点是否投影到墙平面 |
| `depth_wall_noise_k` | 噪声带 = k·r²（相对传感器距离） |
| `depth_wall_noise_max` | 噪声带上限（米） |
| `depth_wall_range_max` | 找墙水平距离上限（米） |
| `depth_wall_ground_margin` | 地面上抬切掉高度（米） |
| `depth_ground_ref_z` | 地面参考高度（米） |
| `wall_fuse_min_support_points` | 多相机融合最小支撑点数 |
| `depth_merge_max_age` | 融合墙结果最大时间差（秒） |
| `depth_wall_tf_report_sec` | TF 失败率日志周期（秒） |

### `wall_track`（深度每相机与雷达共用）

| 参数 | 含义 |
|------|------|
| `normal_gate` | 关联法向距离门限（米） |
| `merge_gap` | 关联沿墙端点间隙允许（米） |
| `obs_weight` | 非 KF 时观测融合权重（0~1） |
| `confirm_frames` | 连续命中多少帧才确认输出 |
| `shorten_frames` | 端点缩短确认帧数（`clamp_extent_to_obs=false` 时） |
| `lose_sec` | 确认后丢检 coast 时长（秒） |
| `coast_gate_extra` | coast 时额外放宽关联门限（米） |
| `clamp_extent_to_obs` | true：匹配更新直接用观测端点，禁止长出本帧支撑 |
| `kf_enable` | 启用 4 维 KF（ρ, θ, 两端弧长） |
| `kf_q_rho` / `kf_q_theta` / `kf_q_end` | 过程噪声 Q（静止墙宜小） |
| `kf_r_rho` / `kf_r_theta` / `kf_r_end` | 观测噪声 R（抖动大宜加大） |

墙静止、墙面跳动大时：可关 `clamp_extent_to_obs`，减小 `kf_q_*`、增大 `kf_r_*`。

### `depth_hough` / `lidar_hough`（公共项）

| 参数 | 含义 |
|------|------|
| `extractor` | `hough` \| `ransac` \| `seed_grow` |
| `resolution` | 占据栅格分辨率（米/格） |
| `half_extent` | 栅格半边长（米），宜 ≥ ROI |
| `max_grid_side` | 栅格单边最大格数 |
| `follow_sensor` | true：栅格中心跟传感器；false：用 `origin_*` |
| `origin_x` / `origin_y` | 固定栅格原点（米） |
| `min_cell_points` | 栅格占用最少点数 |
| `min_cell_height` | 占用最少高度跨度（米，hough） |
| `z_min` / `z_max` | 找墙高度带（米） |
| `theta_deg` / `votes` / `hough_max_gap` | HoughLinesP 角度分辨率、投票、间隙 |
| `min_line_length` | 最短墙段（米） |
| `merge_angle_deg` / `merge_distance` / `merge_gap` | 候选线合并角、法向距、间隙 |
| `parallel_angle_deg` | 小于此夹角视为平行（不做拐角交点，度） |
| `intersection_extension` | 交点允许超出端点的距离（米）；也作 corner_snap 默认距 |
| `support_band` | 支撑点到直线法向带宽（米） |
| `min_support_ratio` | 支撑命中比例下限（0~1） |
| `max_unsupported_gap` | 最大无支撑间隙（米） |
| `min_continuous_support` | 最短连续支撑（米） |
| `min_support_points` | 最少支撑点数 |
| `max_lines` | 单帧最多墙段数 |
| `refine` | 是否用支撑点 TLS 精修 |
| `debug` | 调试开关 |
| `ransac_distance` | RANSAC 内点距（米）；≤0 用 `support_band` |
| `ransac_min_inliers` / `ransac_max_iterations` / `ransac_downsample_leaf` | RANSAC 内点/迭代/降采样 |
| `seed_contour_angle_deg` | seed_grow 方位角分箱（度） |
| `seed_min_points` / `seed_max_distance` | 种子最少点、距离阈值（≤0 用 support_band） |
| `grow_distance` / `grow_max_gap_bins` / `grow_max_range_jump` | 生长距离、空箱数、距离跳变 |

完整默认值与行内中文注释见 `config/wall_plane_filter.yaml`。
