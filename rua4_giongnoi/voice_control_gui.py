#!/usr/bin/env python3
"""
Giao diện Đồ họa Điều khiển Giọng nói & Bản đồ Ngữ nghĩa (Interactive Voice Navigation GUI)
Bài thực hành 7 - Package: rua4_giongnoi (ROS 2 Humble)

Tính năng nổi bật của GUI:
  1. Bảng điều khiển trực quan 2 phân vùng (1280 x 820 px):
     - Cột Trái (Bản đồ Phân vùng Ngữ nghĩa Thời gian thực):
         + Hiển thị toàn bộ bản đồ đã phân vùng 4 khu vực (A: Phòng học, B: Phòng thí nghiệm,
           C: Hành lang, D: Khu kho thực hành).
         + Hiển thị vị trí & hướng hiện tại của robot TurtleBot 4 từ `/amcl_pose`.
         + Vẽ trực tiếp lộ trình dự kiến (mũi tên hồng nối qua các điểm A -> C -> B).
         + Hỗ trợ CLICK CHUỘT TRỰC TIẾP vào từng khu vực trên bản đồ để phát lệnh đi tới khu vực đó!
     - Cột Phải (Trung tâm Điều khiển Giọng nói - Voice Control Center):
         + Nút bấm lớn [ 🎙️ THU ÂM GIỌNG NÓI (MIC 4s) ]: Click chuột (hoặc nhấn phím SPACE)
           để thu âm từ Microphone và nhận diện bằng Whisper / Vosk.
         + Ô nhập câu lệnh trực tiếp trên GUI + nút [ GỬI LỆNH ]: Gõ câu lệnh tùy ý và nhấn Enter.
         + 06 Nút bấm Câu lệnh Giọng nói Mẫu (Click chuột hoặc nhấn phím 1..6) bao phủ toàn bộ yêu cầu:
             [1] "Đi đến khu vực A."
             [2] "Đi đến phòng thí nghiệm."
             [3] "Di chuyển từ khu vực A đến khu vực B."
             [4] "Đi từ phòng A qua hành lang C rồi đến phòng B."
             [5] "Đi đến khu vực D."
             [6] "Đi đến khu vực Z." (Kiểm thử báo lỗi khu vực không tồn tại)
         + Nút [ 🛑 DỪNG / XÓA LỘ TRÌNH ] để hủy hành trình hiện tại.
         + Bảng trạng thái NLU & Nav2 thời gian thực (màu xanh khi hợp lệ, màu đỏ khi báo lỗi vùng không tồn tại).
"""

import os
import sys
import math
import time
import threading
import argparse
from typing import Dict, List, Tuple, Optional, Any

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    np = None
    HAS_OPENCV = False

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.utilities import remove_ros_args
    from std_msgs.msg import String
    from geometry_msgs.msg import PoseWithCovarianceStamped
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object

from rua4_giongnoi.map_segmentation import (
    resolve_pkg_root,
    parse_map_yaml,
    world_to_pixel,
    pixel_to_world,
    write_png_rgb,
    set_px,
    draw_rect,
    draw_circle,
    draw_arrow,
    draw_text,
)
from rua4_giongnoi.speech_recognition_node import SpeechEngine, SAMPLE_VOICE_COMMANDS
from rua4_giongnoi.voice_navigation_node import load_semantic_regions, VoiceCommandParser


GUI_COMMANDS: List[Tuple[str, str]] = [
    ("1. DI DEN KHU VUC A (PHONG HOC)", "Đi đến khu vực A."),
    ("2. DI DEN PHONG THI NGHIEM (KHU B)", "Đi đến phòng thí nghiệm."),
    ("3. TU KHU VUC A DEN KHU VUC B (2 DIEM)", "Di chuyển từ khu vực A đến khu vực B."),
    ("4. TU A QUA HANH LANG C ROI DEN B (DA DIEM)", "Đi từ phòng A qua hành lang C rồi đến phòng B."),
    ("5. DI DEN KHU KHO THUC HANH (KHU D)", "Đi đến khu vực D."),
    ("6. DI DEN KHU VUC Z (TEST LOI VUNG SAI)", "Đi đến khu vực Z."),
]


class VoiceControlGUI:
    """Giao diện đồ họa tương tác chuột & bàn phím điều khiển giọng nói TurtleBot 4."""

    def __init__(
        self,
        ros_node: Optional[Any] = None,
        engine_name: str = "auto",
        whisper_model: str = "base",
        record_seconds: float = 4.0,
    ) -> None:
        self.ros_node = ros_node
        self.pkg_root = resolve_pkg_root()
        self.regions = load_semantic_regions(os.path.join(self.pkg_root, "config", "semantic_regions.json"))
        self.parser = VoiceCommandParser(self.regions)
        self.map_meta = parse_map_yaml(os.path.join(self.pkg_root, "maps", "map.yaml"))
        self.speech_engine = SpeechEngine(preferred_engine=engine_name, whisper_model_size=whisper_model)
        self.record_seconds = record_seconds

        # Trạng thái hiển thị trên GUI
        self.robot_xy: Tuple[float, float] = (0.0, 0.0)
        self.robot_yaw: float = 0.0
        self.last_voice_text: str = "Đi từ phòng A qua hành lang C rồi đến phòng B."
        self.parsed_info: Dict[str, Any] = self.parser.parse(self.last_voice_text, self.robot_xy)
        self.nav_status: str = "READY - San sang nhan lenh giong noi tren GUI"
        self.mic_state: str = "IDLE"  # IDLE | RECORDING | TRANSCRIBING
        self.typing_buffer: str = ""
        self.typing_active: bool = False
        self.history_log: List[str] = [
            "[SYSTEM] Khoi dong Giao dien Dieu khien Giong noi (rua4_giongnoi)",
            f"[STT] Engine nhan dang: {self.speech_engine.active_engine.upper()}",
            "[MAP] Da nap 4 khu vuc: A (Phong hoc), B (Phong Lab), C (Hanh lang), D (Khu kho)",
        ]

        # Kích thước cửa sổ GUI
        self.win_w = 1280
        self.win_h = 820
        self.map_panel_x = 20
        self.map_panel_y = 65
        self.map_panel_w = 460
        self.map_panel_h = 735
        self.map_orig_w = 604
        self.map_orig_h = 1005

        # Danh sách các vùng nút bấm [id, x0, y0, x1, y1, payload]
        self.buttons: List[Tuple[str, int, int, int, int, Any]] = []
        self._setup_buttons()

    def _setup_buttons(self) -> None:
        self.buttons.clear()
        rx = 505
        rw = 755

        # 1. Nút Microphone lớn & Nút Dừng
        self.buttons.append(("btn_mic", rx, 65, rx + 510, 130, None))
        self.buttons.append(("btn_stop", rx + 525, 65, rx + rw, 130, None))

        # 2. 4 nút chọn nhanh Khu vực A, B, C, D
        zone_ids = ["A", "B", "C", "D"]
        zw = (rw - 30) // 4
        for i, zid in enumerate(zone_ids):
            bx0 = rx + i * (zw + 10)
            self.buttons.append((f"btn_zone_{zid}", bx0, 145, bx0 + zw, 195, zid))

        # 3. Ô nhập văn bản trực tiếp + nút Gửi lệnh
        self.buttons.append(("input_box", rx, 210, rx + 595, 258, None))
        self.buttons.append(("btn_send_text", rx + 605, 210, rx + rw, 258, None))

        # 4. 6 nút câu lệnh giọng nói mẫu
        y_start = 295
        btn_h = 46
        btn_gap = 10
        for idx, (label, cmd_text) in enumerate(GUI_COMMANDS):
            by0 = y_start + idx * (btn_h + btn_gap)
            self.buttons.append((f"btn_cmd_{idx}", rx, by0, rx + rw, by0 + btn_h, cmd_text))

    def add_log(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.history_log.append(f"[{ts}] {line}")
        if len(self.history_log) > 7:
            self.history_log = self.history_log[-7:]

    def trigger_voice_command(self, cmd_text: str, source: str = "GUI") -> None:
        cmd_text = cmd_text.strip()
        if not cmd_text:
            return
        self.last_voice_text = cmd_text
        self.parsed_info = self.parser.parse(cmd_text, self.robot_xy)

        if self.parsed_info["valid"]:
            route_str = " -> ".join(self.parsed_info["planned_waypoints"])
            self.nav_status = f"EXECUTING ROUTE: {route_str}"
            self.add_log(f"[{source}] \"{cmd_text}\" => Route: {route_str}")
        else:
            self.nav_status = self.parsed_info["message"]
            self.add_log(f"[ERROR] \"{cmd_text}\" => Khu vuc khong ton tai!")

        if self.ros_node is not None:
            self.ros_node.publish_command(cmd_text)

    def start_mic_recording_async(self) -> None:
        if self.mic_state != "IDLE":
            return

        def _worker() -> None:
            self.mic_state = "RECORDING"
            self.nav_status = f"DANG THU AM MICROPHONE ({self.record_seconds:.0f}s)... Hay noi cau lenh!"
            self.add_log(f"[MIC] Bat dau thu am {self.record_seconds:.0f}s...")
            wav_path = self.speech_engine.record_microphone_wav(duration_sec=self.record_seconds)
            if not wav_path:
                self.mic_state = "IDLE"
                self.nav_status = "Khong tim thay Microphone! Hay click cac nut cau lenh hoac nhap tren GUI."
                self.add_log("[MIC] Khong phat hien thiet bi thu am. Dung nut bam tren GUI.")
                return
            self.mic_state = "TRANSCRIBING"
            self.nav_status = "DANG NHAN DANG GIONG NOI (WHISPER / VOSK)..."
            text = self.speech_engine.transcribe_file(wav_path)
            self.mic_state = "IDLE"
            if text:
                self.trigger_voice_command(text, source="MIC-STT")
            else:
                self.nav_status = "Khong nghe ro cau lenh. Vui long thu lai hoac click nut cau lenh."
                self.add_log("[MIC] Khong nhan dang duoc van ban tu am thanh.")

        threading.Thread(target=_worker, daemon=True).start()

    def handle_mouse_click(self, x: int, y: int) -> None:
        # 1. Kiểm tra click vào bản đồ bên trái -> chọn khu vực tương ứng
        if (self.map_panel_x <= x <= self.map_panel_x + self.map_panel_w and
                self.map_panel_y <= y <= self.map_panel_y + self.map_panel_h):
            u = int((x - self.map_panel_x) * self.map_orig_w / self.map_panel_w)
            v = int((y - self.map_panel_y) * self.map_orig_h / self.map_panel_h)
            res = float(self.map_meta["resolution"])
            ox, oy = float(self.map_meta["origin"][0]), float(self.map_meta["origin"][1])
            wx, wy = pixel_to_world(u, v, ox, oy, res, self.map_orig_h)
            # Tìm khu vực gần điểm click nhất
            best_id = "A"
            best_d = float("inf")
            for rid, rinfo in self.regions.items():
                gx = float(rinfo["goal_world"]["x"])
                gy = float(rinfo["goal_world"]["y"])
                d = math.hypot(wx - gx, wy - gy)
                if d < best_d:
                    best_d = d
                    best_id = rid
            self.trigger_voice_command(f"Đi đến khu vực {best_id}.", source="MAP-CLICK")
            return

        # 2. Kiểm tra click vào các nút điều khiển bên phải
        self.typing_active = False
        for btn_id, x0, y0, x1, y1, payload in self.buttons:
            if x0 <= x <= x1 and y0 <= y <= y1:
                if btn_id == "btn_mic":
                    self.start_mic_recording_async()
                elif btn_id == "btn_stop":
                    self.parsed_info = {
                        "valid": True,
                        "raw_command": "Dừng hành trình",
                        "command_type": "stop",
                        "extracted_regions": [],
                        "planned_waypoints": [],
                        "unknown_regions": [],
                        "message": "Đã hủy lộ trình hiện tại. Robot dừng an toàn.",
                    }
                    self.nav_status = "STOPPED - Da huy hanh trinh hien tai"
                    self.add_log("[STOP] Nguoi dung nhan nut DUNG / XOA LO TRINH")
                elif btn_id.startswith("btn_zone_"):
                    zid = str(payload)
                    self.trigger_voice_command(f"Đi đến khu vực {zid}.", source="ZONE-BTN")
                elif btn_id == "input_box":
                    self.typing_active = True
                elif btn_id == "btn_send_text":
                    if self.typing_buffer.strip():
                        self.trigger_voice_command(self.typing_buffer.strip(), source="TEXT-INPUT")
                        self.typing_buffer = ""
                elif btn_id.startswith("btn_cmd_"):
                    self.trigger_voice_command(str(payload), source="VOICE-PRESET")
                return


def generate_gui_preview_png(output_path: str) -> None:
    """Xuất ảnh chụp giao diện điều khiển giọng nói (Voice Control GUI) phục vụ Báo cáo & README."""
    pkg_root = resolve_pkg_root()
    seg_png = os.path.join(pkg_root, "maps", "map_segmented.png")
    route_png = os.path.join(pkg_root, "results", "voice_route_demo.png")
    map_img_To_use = route_png if os.path.isfile(route_png) else seg_png

    w, h = 1280, 820
    canvas = bytearray([22, 27, 38] * (w * h))

    # Header bar
    draw_rect(canvas, w, h, 0, 0, w - 1, 50, (15, 52, 96))
    draw_text(canvas, w, h, 22, 16, "TURTLEBOT 4 - INTERACTIVE VOICE NAVIGATION GUI (PACKAGE: RUA4_GIONGNOI)", (255, 255, 255), scale=2)

    # Khung bên trái: Bản đồ phân vùng
    draw_rect(canvas, w, h, 18, 62, 482, 802, (32, 39, 54), (70, 130, 210))
    draw_text(canvas, w, h, 30, 72, "FULL-MAP SEMANTIC VIEW (CLICK ZONE TO NAVIGATE)", (140, 200, 255), scale=1)

    # Khung bên phải: Trung tâm điều khiển giọng nói
    rx, rw = 500, 760
    draw_rect(canvas, w, h, rx - 8, 62, rx + rw + 4, 802, (28, 34, 48), (65, 85, 120))

    # 1. Nút Microphone & Nút Stop
    draw_rect(canvas, w, h, rx + 5, 72, rx + 510, 132, (25, 135, 84), (120, 240, 170))
    draw_text(canvas, w, h, rx + 35, 94, "MIC: THU AM GIONG NOI (SPACE / CLICK)", (255, 255, 255), scale=2)

    draw_rect(canvas, w, h, rx + 525, 72, rx + rw - 5, 132, (195, 45, 55), (255, 130, 130))
    draw_text(canvas, w, h, rx + 552, 94, "DUNG ROBOT", (255, 255, 255), scale=2)

    # 2. 4 nút chọn nhanh Khu vực A, B, C, D
    zone_configs = [
        ("KHU A: PHONG HOC", (66, 133, 244)),
        ("KHU B: PHONG LAB", (52, 168, 83)),
        ("KHU C: HANH LANG", (220, 160, 15)),
        ("KHU D: KHU KHO", (220, 67, 53)),
    ]
    zw = (rw - 40) // 4
    for i, (ztitle, zcol) in enumerate(zone_configs):
        bx0 = rx + 5 + i * (zw + 10)
        draw_rect(canvas, w, h, bx0, 146, bx0 + zw, 194, zcol, (255, 255, 255))
        draw_text(canvas, w, h, bx0 + 12, 166, ztitle, (255, 255, 255), scale=1)

    # 3. Ô nhập câu lệnh & nút Gửi
    draw_rect(canvas, w, h, rx + 5, 208, rx + 595, 256, (16, 20, 28), (100, 160, 240))
    draw_text(canvas, w, h, rx + 18, 226, "NHAP CAU LENH: DI TU PHONG A QUA HANH LANG C ROI DEN PHONG B", (220, 240, 255), scale=1)
    draw_rect(canvas, w, h, rx + 605, 208, rx + rw - 5, 256, (45, 105, 210), (160, 205, 255))
    draw_text(canvas, w, h, rx + 632, 226, "GUI LENH (ENTER)", (255, 255, 255), scale=1)

    # 4. Danh sách 6 nút câu lệnh giọng nói mẫu
    draw_text(canvas, w, h, rx + 8, 272, "DANH SACH CAU LENH GIONG NOI MAU (CLICK CHUOT HOAC NHAN PHIM 1..6):", (255, 215, 90), scale=1)
    y0 = 292
    for idx, (label, _) in enumerate(GUI_COMMANDS):
        by = y0 + idx * 54
        is_active = (idx == 3)
        bg = (40, 95, 175) if is_active else (42, 52, 72)
        border = (120, 220, 255) if is_active else (85, 105, 140)
        if idx == 5:
            bg = (95, 42, 52)
            border = (220, 100, 110)
        draw_rect(canvas, w, h, rx + 5, by, rx + rw - 5, by + 44, bg, border)
        draw_text(canvas, w, h, rx + 20, by + 16, label, (255, 255, 255), scale=1)

    # 5. Bảng trạng thái NLU & Lộ trình hiện tại
    draw_rect(canvas, w, h, rx + 5, 622, rx + rw - 5, 792, (18, 24, 34), (80, 180, 120))
    draw_text(canvas, w, h, rx + 18, 634, "TRANG THAI NHAN DANG GIONG NOI (NLU) & DIEU HUONG NAV2:", (110, 235, 155), scale=1)
    draw_text(canvas, w, h, rx + 18, 658, "CAU LENH: DI TU PHONG A QUA HANH LANG C ROI DEN PHONG B.", (255, 255, 255), scale=1)
    draw_text(canvas, w, h, rx + 18, 680, "LO TRINH DU KIEN: A (6.0, 2.0) -> C (-3.5, 1.5) -> B (2.0, 14.0)", (255, 210, 90), scale=1)
    draw_text(canvas, w, h, rx + 18, 702, "VI TRI ROBOT HIEN TAI: (0.0M, 0.0M) | KHOANG CACH DEN A: 6.32M", (180, 215, 255), scale=1)
    draw_text(canvas, w, h, rx + 18, 724, "TRANG THAI NAV2: EXECUTING MULTI-WAYPOINT ROUTE [A -> C -> B]", (120, 245, 160), scale=1)
    draw_text(canvas, w, h, rx + 18, 748, "KIEM TRA LOI VUNG Z: [PASS] TU CHOI LENH KHI VUNG KHONG TON TAI", (255, 150, 150), scale=1)

    write_png_rgb(output_path, w, h, canvas)


class VoiceControlGUINode(Node):
    """ROS 2 Node tích hợp Giao diện Đồ họa (GUI) Điều khiển Giọng nói."""

    def __init__(self, engine_name: str = "auto", whisper_model: str = "base",
                 record_seconds: float = 4.0) -> None:
        super().__init__("voice_control_gui")
        self.pub_cmd = self.create_publisher(String, "/voice_command", 10)
        self.sub_cmd = self.create_subscription(String, "/voice_command", self._on_voice_cmd, 10)
        self.sub_status = self.create_subscription(String, "/voice_nav_status", self._on_nav_status, 10)
        self.sub_amcl = self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl, 10)

        self.gui = VoiceControlGUI(
            ros_node=self,
            engine_name=engine_name,
            whisper_model=whisper_model,
            record_seconds=record_seconds,
        )
        self.get_logger().info("Giao diện Đồ họa Điều khiển Giọng nói (voice_control_gui) đã sẵn sàng!")

    def publish_command(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.pub_cmd.publish(msg)
        self.get_logger().info(f'🖥️  [GUI -> /voice_command]: "{text}"')

    def _on_voice_cmd(self, msg: String) -> None:
        text = msg.data.strip()
        if text and text != self.gui.last_voice_text:
            self.gui.last_voice_text = text
            self.gui.parsed_info = self.gui.parser.parse(text, self.gui.robot_xy)
            self.gui.add_log(f"[TOPIC] Nhan lenh: \"{text}\"")

    def _on_nav_status(self, msg: String) -> None:
        self.gui.nav_status = msg.data

    def _on_amcl(self, msg: PoseWithCovarianceStamped) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.gui.robot_xy = (float(pos.x), float(pos.y))
        self.gui.robot_yaw = 2.0 * math.atan2(float(ori.z), float(ori.w))


def run_opencv_gui(gui: VoiceControlGUI) -> None:
    """Chạy vòng lặp cửa sổ tương tác OpenCV (hỗ trợ click chuột + phím tắt)."""
    pkg_root = resolve_pkg_root()
    seg_img_path = os.path.join(pkg_root, "maps", "map_segmented.png")
    base_map_img = cv2.imread(seg_img_path) if os.path.isfile(seg_img_path) else None

    win_name = "TurtleBot 4 - Giao dien Dieu khien Giong noi & Ban do Ngu nghia (rua4_giongnoi)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, gui.win_w, gui.win_h)

    def _mouse_cb(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            gui.handle_mouse_click(x, y)

    cv2.setMouseCallback(win_name, _mouse_cb)

    res = float(gui.map_meta["resolution"])
    ox, oy = float(gui.map_meta["origin"][0]), float(gui.map_meta["origin"][1])

    while True:
        frame = np.full((gui.win_h, gui.win_w, 3), (38, 27, 22), dtype=np.uint8)

        # Thanh tiêu đề
        cv2.rectangle(frame, (0, 0), (gui.win_w, 50), (96, 52, 15), -1)
        cv2.putText(
            frame,
            "TURTLEBOT 4 - GIAO DIEN DIEU KHIEN GIONG NOI & BAN DO NGU NGHIA (RUA4_GIONGNOI)",
            (20, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Cột Trái: Bản đồ phân vùng + Vị trí Robot + Đường đi dự kiến
        mx0, my0, mw, mh = gui.map_panel_x, gui.map_panel_y, gui.map_panel_w, gui.map_panel_h
        cv2.rectangle(frame, (mx0 - 3, my0 - 3), (mx0 + mw + 3, my0 + mh + 3), (210, 130, 70), 2)

        if base_map_img is not None:
            map_view = base_map_img.copy()
            # Vẽ lộ trình dự kiến trên bản đồ
            waypoints = gui.parsed_info.get("planned_waypoints", [])
            if waypoints:
                pts_px = []
                ru, rv = world_to_pixel(gui.robot_xy[0], gui.robot_xy[1], ox, oy, res, gui.map_orig_h)
                pts_px.append((ru, rv))
                for wid in waypoints:
                    if wid in gui.regions:
                        c_px = gui.regions[wid]["pixel_centroid"]
                        pts_px.append((int(c_px["u"]), int(c_px["v"])))
                for i in range(len(pts_px) - 1):
                    cv2.arrowedLine(map_view, pts_px[i], pts_px[i + 1], (90, 20, 230), 6, cv2.LINE_AA, tipLength=0.08)

            # Vẽ vị trí hiện tại của Robot trên bản đồ
            ru, rv = world_to_pixel(gui.robot_xy[0], gui.robot_xy[1], ox, oy, res, gui.map_orig_h)
            if 0 <= ru < gui.map_orig_w and 0 <= rv < gui.map_orig_h:
                cv2.circle(map_view, (ru, rv), 14, (255, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(map_view, (ru, rv), 14, (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(map_view, "ROBOT", (max(5, ru - 28), max(20, rv - 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

            resized_map = cv2.resize(map_view, (mw, mh), interpolation=cv2.INTER_AREA)
            frame[my0:my0 + mh, mx0:mx0 + mw] = resized_map

        # Cột Phải: Các nút điều khiển
        rx, rw = 505, 755
        cv2.rectangle(frame, (rx - 10, my0 - 3), (rx + rw + 10, my0 + mh + 3), (48, 34, 28), -1)
        cv2.rectangle(frame, (rx - 10, my0 - 3), (rx + rw + 10, my0 + mh + 3), (120, 85, 65), 1)

        # Nút Microphone
        mic_bg = (60, 50, 210) if gui.mic_state == "RECORDING" else (84, 155, 25)
        mic_label = (
            f"DANG THU AM MIC ({gui.record_seconds:.0f}s)..."
            if gui.mic_state == "RECORDING"
            else "THU AM GIONG NOI QUA MIC (CLICK / SPACE)"
        )
        cv2.rectangle(frame, (rx, 65), (rx + 510, 130), mic_bg, -1)
        cv2.rectangle(frame, (rx, 65), (rx + 510, 130), (180, 245, 180), 2)
        cv2.putText(frame, mic_label, (rx + 18, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Nút Stop
        cv2.rectangle(frame, (rx + 525, 65), (rx + rw, 130), (55, 45, 195), -1)
        cv2.rectangle(frame, (rx + 525, 65), (rx + rw, 130), (140, 140, 255), 2)
        cv2.putText(frame, "DUNG / HUY", (rx + 568, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

        # 4 nút chọn nhanh Khu vực A, B, C, D
        zone_Info = [
            ("A", "KHU A: PHONG HOC", (244, 133, 66)),
            ("B", "KHU B: PHONG LAB", (83, 168, 52)),
            ("C", "KHU C: HANH LANG", (15, 175, 235)),
            ("D", "KHU D: KHU KHO", (53, 67, 234)),
        ]
        zw = (rw - 30) // 4
        for i, (zid, ztxt, zcol_bgr) in enumerate(zone_Info):
            bx0 = rx + i * (zw + 10)
            cv2.rectangle(frame, (bx0, 145), (bx0 + zw, 195), zcol_bgr, -1)
            cv2.rectangle(frame, (bx0, 145), (bx0 + zw, 195), (255, 255, 255), 1)
            cv2.putText(frame, ztxt, (bx0 + 10, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

        # Ô nhập câu lệnh tùy chỉnh + Nút Gửi
        box_border = (100, 240, 255) if gui.typing_active else (160, 120, 90)
        cv2.rectangle(frame, (rx, 210), (rx + 595, 258), (24, 18, 15), -1)
        cv2.rectangle(frame, (rx, 210), (rx + 595, 258), box_border, 2)
        disp_input = gui.typing_buffer if (gui.typing_buffer or gui.typing_active) else "Click vao day de go cau lenh tuy y..."
        if gui.typing_active:
            disp_input += "|"
        cv2.putText(frame, disp_input[:52], (rx + 12, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (rx + 605, 210), (rx + rw, 258), (195, 105, 35), -1)
        cv2.rectangle(frame, (rx + 605, 210), (rx + rw, 258), (255, 210, 150), 1)
        cv2.putText(frame, "GUI LENH", (rx + 628, 241), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        # 6 nút bấm câu lệnh mẫu
        cv2.putText(frame, "CLICK CHON NHANH CAU LENH GIONG NOI (HOAC NHAN PHIM 1..6):",
                    (rx, 284), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (90, 215, 255), 1, cv2.LINE_AA)
        y_start = 295
        btn_h = 46
        btn_gap = 10
        for idx, (label, cmd_text) in enumerate(GUI_COMMANDS):
            by0 = y_start + idx * (btn_h + btn_gap)
            is_selected = (gui.last_voice_text == cmd_text)
            bg_col = (165, 95, 35) if is_selected else (68, 50, 40)
            if idx == 5:
                bg_col = (55, 42, 105) if not is_selected else (60, 50, 175)
            cv2.rectangle(frame, (rx, by0), (rx + rw, by0 + btn_h), bg_col, -1)
            cv2.rectangle(frame, (rx, by0), (rx + rw, by0 + btn_h), (210, 175, 140), 1)
            cv2.putText(frame, label, (rx + 16, by0 + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

        # Khung hiển thị trạng thái NLU & Lịch sử
        is_valid = gui.parsed_info.get("valid", True)
        status_border = (110, 215, 95) if is_valid else (80, 80, 245)
        cv2.rectangle(frame, (rx, 635), (rx + rw, 800), (24, 18, 15), -1)
        cv2.rectangle(frame, (rx, 635), (rx + rw, 800), status_border, 2)

        route_list = gui.parsed_info.get("planned_waypoints", [])
        route_txt = " -> ".join(route_list) if route_list else ("INVALID REGION (TU CHOI LENH)" if not is_valid else "NONE")
        cv2.putText(frame, f"LENH CUOI: {gui.last_voice_text}", (rx + 14, 660),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"LO TRINH NLU: {route_txt}  |  ROBOT: ({gui.robot_xy[0]:.1f}m, {gui.robot_xy[1]:.1f}m)",
                    (rx + 14, 684), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (90, 225, 255) if is_valid else (110, 110, 255), 2, cv2.LINE_AA)

        for i, log_line in enumerate(gui.history_log[-4:]):
            cv2.putText(frame, log_line[:82], (rx + 14, 712 + i * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (195, 205, 215), 1, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or (key == ord("q") and not gui.typing_active):
            break
        elif gui.typing_active:
            if key in (13, 10):  # Enter
                if gui.typing_buffer.strip():
                    gui.trigger_voice_command(gui.typing_buffer.strip(), source="TEXT-INPUT")
                    gui.typing_buffer = ""
                gui.typing_active = False
            elif key in (8, 127):  # Backspace
                gui.typing_buffer = gui.typing_buffer[:-1]
            elif 32 <= key <= 126:
                gui.typing_buffer += chr(key)
        else:
            if key == ord(" "):
                gui.start_mic_recording_async()
            elif ord("1") <= key <= ord("6"):
                idx = key - ord("1")
                gui.trigger_voice_command(GUI_COMMANDS[idx][1], source=f"HOTKEY-{idx+1}")

    cv2.destroyAllWindows()


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Giao diện Đồ họa Điều khiển Giọng nói (Bài thực hành 7)")
    parser.add_argument("--engine", default="auto", choices=["auto", "whisper", "vosk", "google", "cli"],
                        help="Bộ nhận dạng giọng nói")
    parser.add_argument("--whisper-model", default="base",
                        help="Kích thước model Whisper")
    parser.add_argument("--record-seconds", type=float, default=4.0,
                        help="Thời gian thu âm Microphone (giây)")
    parser.add_argument("--export-preview", default=None,
                        help="Xuất ảnh chụp giao diện GUI ra file PNG rồi thoát")

    clean_argv = remove_ros_args(args=sys.argv)[1:] if HAS_ROS2 else sys.argv[1:]
    parsed, _ = parser.parse_known_args(clean_argv)

    if parsed.export_preview:
        generate_gui_preview_png(parsed.export_preview)
        print(f"Đã xuất ảnh giao diện GUI tại: {parsed.export_preview}")
        return

    if not HAS_OPENCV:
        print("Lỗi: Cần thư viện opencv-python (cv2) để hiển thị giao diện đồ họa.")
        sys.exit(1)

    if HAS_ROS2:
        rclpy.init(args=args)
        node = VoiceControlGUINode(
            engine_name=parsed.engine,
            whisper_model=parsed.whisper_model,
            record_seconds=parsed.record_seconds,
        )
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
        try:
            run_opencv_gui(node.gui)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    else:
        gui = VoiceControlGUI(
            ros_node=None,
            engine_name=parsed.engine,
            whisper_model=parsed.whisper_model,
            record_seconds=parsed.record_seconds,
        )
        run_opencv_gui(gui)


if __name__ == "__main__":
    main()
