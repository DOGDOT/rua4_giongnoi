# Package `rua4_giongnoi` — Phân vùng Bản đồ & Điều hướng Robot TurtleBot 4 bằng Giọng nói (Bài thực hành 7)

Package **ROS 2 Humble** hoạt động **hoàn toàn độc lập** cho robot **TurtleBot 4** trong môi trường mô phỏng **Gazebo Sim (Ignition)**, giải quyết trọn vẹn các yêu cầu của **Bài thực hành 7: Phân vùng bản đồ và điều hướng robot bằng giọng nói** (kèm phần mở rộng điều hướng đa điểm):

1. **Phân vùng bản đồ (Map Segmentation)**:
   - Xử lý bản đồ 2D Occupancy Grid (`maps/map.pgm` & `maps/map.yaml`) bằng **OpenCV** (*Binary Thresholding* $\rightarrow$ *Morphological Filtering* $\rightarrow$ *Distance Transform* $\rightarrow$ *Seeded Watershed / Voronoi Partitioning*).
   - Xuất bản đồ phân vùng trực quan `maps/map_segmented.png` và ảnh chuỗi xử lý 4 bước `results/map_segmentation_pipeline.png`.
2. **Gắn nhãn khu vực ngữ nghĩa (Semantic Mapping)**:
   - Phân chia và hiển thị trực quan **04 khu vực có nhãn** (vượt yêu cầu tối thiểu 3 khu vực), lưu thông tin tọa độ pixel & tọa độ thực $(x, y, \text{yaw})$ trong hệ tọa độ `map` dưới cả 2 định dạng `config/semantic_regions.yaml` và `config/semantic_regions.json`.
   - Hiển thị trực tiếp các khu vực màu, nhãn 3D và đường đi dự kiến trên **RViz2** qua topic `/semantic_regions_markers` (`visualization_msgs/msg/MarkerArray`).
3. **Nhận dạng giọng nói (Speech Recognition — Whisper / Vosk)**:
   - Node `speech_recognition_node` tiếp nhận câu lệnh giọng nói tiếng Việt / tiếng Anh qua Microphone (hỗ trợ **OpenAI Whisper**, **Vosk Offline**, hoặc bảng điều khiển tương tác nhanh) và publish văn bản lên `/voice_command` (`std_msgs/msg/String`).
4. **Điều hướng bằng giọng nói & Lộ trình Đa điểm (Voice Navigation & Multi-Waypoint Nav2)**:
   - Node `voice_navigation_node` phân tích câu lệnh, tự động kiểm tra vị trí hiện tại của robot để lập chuỗi mục tiêu `NavigateToPose` cho Nav2:
     - **Lệnh đơn điểm**: *"Đi đến khu vực A."*, *"Đi đến phòng thí nghiệm."*
     - **Lệnh 2 điểm (Từ A đến B)**: *"Di chuyển từ khu vực A đến khu vực B."* $\rightarrow$ Nếu robot chưa ở A, tự động đi **qua A trước khi đến B**.
     - **Lệnh mở rộng đa điểm**: *"Đi từ phòng A qua hành lang C rồi đến phòng B."* $\rightarrow$ Tự động đi tuần tự `A → C → B`.
     - **Bắt lỗi khu vực không tồn tại**: *"Đi đến khu vực Z."*, *"Đi đến phòng giám đốc."* $\rightarrow$ Phát hiện vùng không tồn tại, cảnh báo rõ ràng và giữ robot đứng yên an toàn.

---

## 📁 Cấu trúc Thư mục Package

```text
rua4_giongnoi/
├── package.xml                          # Khai báo dependencies (rclpy, std_msgs, visualization_msgs, nav2...)
├── setup.py                             # Cấu hình cài đặt Python package & 4 console_scripts
├── setup.cfg
├── rua4_giongnoi/                       # Mã nguồn chính của package
│   ├── __init__.py
│   ├── map_segmentation.py              # Phân vùng bản đồ .pgm bằng OpenCV & xuất YAML/JSON + ảnh PNG
│   ├── speech_recognition_node.py       # Node nhận dạng giọng nói (Whisper / Vosk / Interactive CLI)
│   ├── voice_navigation_node.py         # Node phân tích câu lệnh NLU, lập lộ trình đa điểm & gửi Goal Nav2
│   └── semantic_map_viewer.py           # Hiển thị cửa sổ bản đồ phân vùng ngữ nghĩa bằng OpenCV
├── launch/                              # Các file launch khởi chạy hệ thống độc lập
│   ├── sim.launch.py                    # Khởi chạy Gazebo + Spawn TurtleBot 4
│   ├── nav2.launch.py                   # Khởi chạy Localization (AMCL) + Nav2 + RViz2
│   ├── sim_nav2.launch.py               # Khởi chạy gộp Gazebo + Localization + Nav2 + RViz2
│   ├── lab7_voice_nav.launch.py         # Khởi chạy đồng thời speech_recognition_node + voice_navigation_node
│   ├── turtlebot4_spawn.launch.py
│   └── include/                         # Cấu hình controller Create 3 & TF
├── config/                              # Cấu hình tham số & dữ liệu vùng ngữ nghĩa
│   ├── semantic_regions.yaml            # Thông tin 4 khu vực ngữ nghĩa (ID, tên, pixel, tọa độ map, bán kính)
│   ├── semantic_regions.json            # Thông tin 4 khu vực ngữ nghĩa ở định dạng JSON
│   ├── localization.yaml                # Cấu hình AMCL Particle Filter
│   ├── nav2.yaml                        # Cấu hình Nav2 Planner, Controller, Costmaps
│   └── rviz/
│       ├── rua4_nav2.rviz               # Giao diện RViz2 đã tích hợp sẵn MarkerArray /semantic_regions_markers
│       └── rua4_sim.rviz
├── maps/                                # Bản đồ gốc và bản đồ đã phân vùng
│   ├── map.pgm                          # Bản đồ Occupancy Grid gốc (604x1005 px, resolution 0.05m)
│   ├── map.yaml                         # Metadata bản đồ gốc (origin: [-15.1, -25.0, 0.0])
│   ├── map.png                          # Ảnh bản đồ gốc định dạng PNG
│   └── map_segmented.png                # Bản đồ phân vùng trực quan 4 khu vực A, B, C, D
└── results/                             # Kết quả thực nghiệm & kiểm thử tự động
    ├── map_segmentation_pipeline.png    # Chuỗi 4 bước xử lý phân vùng bản đồ bằng OpenCV
    ├── voice_route_demo.png             # Minh họa lộ trình đa điểm (D -> A -> C -> B) trên bản đồ phân vùng
    └── voice_navigation_test_report.txt # Báo cáo kết quả kiểm thử tự động 7/7 kịch bản câu lệnh
```

---

## 🗺️ Bảng Khu vực Ngữ nghĩa trên Bản đồ (`semantic_regions.yaml`)

Công thức chuyển đổi từ tọa độ pixel $(u, v)$ trên ảnh `map.pgm` ($W = 604, H = 1005, r = 0.05\,\text{m/px}, x_0 = -15.1, y_0 = -25.0$) sang tọa độ thực $(x, y)$ trong frame `map` của ROS 2:
$$x = x_0 + u \cdot r, \qquad y = y_0 + (H - 1 - v) \cdot r$$

| Mã vùng | Tên Khu vực Ngữ nghĩa | Tọa độ Pixel $(u, v)$ | Tọa độ Đích ROS `map` $(x, y, \text{yaw})$ | Diện tích vùng | Các cụm từ nhận diện giọng nói (Aliases) |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **`A`** | **Khu vực A – Phòng học** | `(422, 464)` | `(6.0, 2.0, 0.00 rad)` | `166.05 m²` | *"khu vực A"*, *"phòng A"*, *"phòng học"*, *"classroom"* |
| **`B`** | **Khu vực B – Phòng thí nghiệm** | `(342, 224)` | `(2.0, 14.0, 1.57 rad)` | `366.84 m²` | *"khu vực B"*, *"phòng B"*, *"phòng thí nghiệm"*, *"phòng lab"* |
| **`C`** | **Khu vực C – Hành lang** | `(232, 474)` | `(-3.5, 1.5, 3.14 rad)` | `222.07 m²` | *"khu vực C"*, *"hành lang"*, *"hành lang C"*, *"corridor"* |
| **`D`** | **Khu vực D – Khu kho thực hành** | `(342, 664)` | `(2.0, -8.0, -1.57 rad)` | `430.68 m²` | *"khu vực D"*, *"trạm xuất phát"*, *"điểm xuất phát"*, *"trạm sạc"* |

---

## 🛠️ Yêu cầu Môi trường & Hướng dẫn Build

### 1. Yêu cầu hệ thống
- **Hệ điều hành**: Ubuntu 22.04 LTS *(Ghi chú: Nếu dùng bản phân phối Linux khác, có thể khởi chạy môi trường Ubuntu 22.04 trước khi chạy các lệnh bên dưới)*.
- **Middleware**: ROS 2 Humble Hawksbill (`turtlebot4_ignition_bringup`, `turtlebot4_navigation`, `nav2_simple_commander`).
- **Thư viện Python**:
  ```bash
  pip install "numpy<2" opencv-python faster-whisper vosk SpeechRecognition
  ```
  *(Ghi chú: Nếu chưa cài `faster-whisper` hoặc `vosk`, node vẫn hoạt động mượt mà với chế độ chọn câu lệnh giọng nói mẫu / nhập lệnh trực tiếp trên Terminal).*

### 2. Build Package `rua4_giongnoi`
```bash
cd ~/ur_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select rua4_giongnoi
source install/setup.bash
```

---

## 🚀 Hướng dẫn Thực hiện Chi tiết Bài Thực Hành 7

### Bước 1 – Chạy Phân vùng Bản đồ & Xuất File Ngữ nghĩa (`map_segmenter`)
```bash
cd ~/ur_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

ros2 run rua4_giongnoi map_segmenter --show
```
- Lệnh trên đọc `maps/map.pgm` & `maps/map.yaml`, thực hiện phân vùng và tự động cập nhật:
  - `maps/map_segmented.png`
  - `results/map_segmentation_pipeline.png`
  - `config/semantic_regions.yaml` & `config/semantic_regions.json`

### Bước 2 – Khởi chạy Mô phỏng Gazebo & Hệ thống Điều hướng Nav2 (Terminal 1 & 2)
- **Terminal 1 (Khởi chạy Gazebo + TurtleBot 4)**:
  ```bash
  cd ~/ur_ws
  source /opt/ros/humble/setup.bash && source install/setup.bash

  ros2 launch rua4_giongnoi sim.launch.py world:=warehouse
  ```
- **Terminal 2 (Khởi chạy Localization AMCL + Nav2 + RViz2)**:
  ```bash
  cd ~/ur_ws
  source /opt/ros/humble/setup.bash && source install/setup.bash

  ros2 launch rua4_giongnoi nav2.launch.py
  ```
  *(Trên RViz2, mục **Semantic Regions & Voice Route** (`/semantic_regions_markers`) đã được bật sẵn, hiển thị trực quan 4 vùng màu A, B, C, D kèm nhãn tên 3D trên bản đồ).*

### Bước 3 – Khởi chạy Nhận dạng Giọng nói & Điều hướng Bằng Giọng nói (Terminal 3)
Mở **Terminal 3**:
```bash
cd ~/ur_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

ros2 launch rua4_giongnoi lab7_voice_nav.launch.py
```
*(Hoặc chạy tách riêng `ros2 run rua4_giongnoi voice_navigation_node` ở Terminal 3 và `ros2 run rua4_giongnoi speech_recognition_node` ở Terminal 4).*

### Bước 4 – Kịch bản Thử nghiệm & Quay Video Demo
Tại cửa sổ Terminal của `speech_recognition_node`, bạn có thể **nhấn `Enter` để nói trực tiếp vào Microphone**, hoặc **nhấn phím `1` – `6`** (hoặc phát qua topic `/voice_command`) để kiểm chứng đầy đủ các yêu cầu:

1. **Câu lệnh 1 (Đơn điểm theo mã vùng)**:
   - Nói / Chọn `[1]`: **`"Đi đến khu vực A."`**
   - $\rightarrow$ Robot xác định đích **Khu vực A – Phòng học** `(6.0, 2.0)` và tự động di chuyển tránh vật cản tới A.
2. **Câu lệnh 2 (Đơn điểm theo tên ngữ nghĩa)**:
   - Nói / Chọn `[2]`: **`"Đi đến phòng thí nghiệm."`**
   - $\rightarrow$ Robot ánh xạ *"phòng thí nghiệm"* sang **Khu vực B** `(2.0, 14.0)` và điều hướng tới B.
3. **Câu lệnh 3 (Đi từ A đến B khi robot đang ở vị trí khác)**:
   - Nói / Chọn `[3]`: **`"Di chuyển từ khu vực A đến khu vực B."`**
   - $\rightarrow$ Hệ thống phát hiện robot chưa đứng ở A nên tự động lập lộ trình **qua A trước rồi mới đến B** (`A → B`).
4. **Câu lệnh 4 (Mở rộng đa điểm — Mục 5)**:
   - Nói / Chọn `[4]`: **`"Đi từ phòng A qua hành lang C rồi đến phòng B."`**
   - $\rightarrow$ Hệ thống lập chuỗi hành trình 3 chặng liên tiếp **`A → C → B`**, vẽ đường dự kiến màu hồng trên RViz2 và lần lượt đi qua từng khu vực.
5. **Câu lệnh 5 (Xử lý khu vực không tồn tại — Yêu cầu 3.4)**:
   - Nói / Chọn `[5]`: **`"Đi đến khu vực Z."`** (hoặc *"Đi đến phòng giám đốc."*)
   - $\rightarrow$ Hệ thống báo lỗi `[LỖI KHU VỰC KHÔNG TỒN TẠI] Không tìm thấy 'Khu vực Z' trên bản đồ!`, từ chối gửi Goal sai và giữ robot đứng yên an toàn.

---

## 🧪 Chạy Kiểm thử Tự động Bộ Phân tích Câu lệnh (Self-Test)

Để kiểm tra nhanh toàn bộ 7 kịch bản phân tích câu lệnh giọng nói mà không cần bật Gazebo:
```bash
ros2 run rua4_giongnoi voice_navigation_node --self-test
```

---

## 🔧 Xử lý Lỗi Thường Gặp (Troubleshooting)

1. **Lỗi `TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'` khi `colcon build`**:
   ```bash
   pip uninstall -y setuptools
   colcon build --symlink-install --packages-select rua4_giongnoi
   ```
2. **Lỗi `AttributeError: _ARRAY_API not found` khi import OpenCV / cv_bridge**:
   ```bash
   pip install "numpy<2"
   ```
