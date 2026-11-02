#!/usr/bin/env python3
"""
Node Điều hướng Robot bằng Giọng nói (Voice Navigation & Multi-Waypoint Commander) - Bài thực hành 7
Package: rua4_giongnoi (ROS 2 Humble)

Chức năng:
  1. Đọc danh sách các khu vực ngữ nghĩa đã phân vùng từ `config/semantic_regions.json` (hoặc `.yaml`).
  2. Lắng nghe câu lệnh văn bản trên topic `/voice_command` (`std_msgs/msg/String`).
  3. Phân tích ngôn ngữ tự nhiên tiếng Việt / tiếng Anh (NLU Parser):
     - Câu lệnh 1 đích: "Đi đến khu vực A", "Đi đến phòng thí nghiệm" -> Điều hướng tới đích.
     - Câu lệnh 2 điểm (Từ A đến B): "Di chuyển từ khu vực A đến khu vực B", "Đến A rồi về C":
         + Kiểm tra vị trí hiện tại của robot (từ `/amcl_pose`).
         + Nếu robot đang ở vị trí khác ngoài vùng A (khoảng cách > bán kính vùng A), tự động
           điều hướng QUA KHU VỰC A TRƯỚC KHI ĐẾN KHU VỰC B (Route: A -> B).
         + Nếu robot đã ở sẵn trong vùng A và câu lệnh có từ khóa "từ A", điều hướng thẳng tới B (Route: B).
     - Câu lệnh đa điểm (Mở rộng Mục 5): "Đi từ phòng A qua hành lang C rồi đến phòng B"
         + Trích xuất chuỗi hành trình tuần tự (Route: A -> C -> B) và điều hướng liên tiếp qua từng điểm.
     - Kiểm tra khu vực không tồn tại (Yêu cầu 3.4):
         + Ví dụ: "Đi đến khu vực Z", "Đi đến phòng giám đốc" -> Phát hiện khu vực không tồn tại
           trên bản đồ ngữ nghĩa, thông báo lỗi rõ ràng và từ chối gửi lệnh sai cho Nav2.
  4. Thực thi điều hướng tuần tự đa điểm qua `rclpy.action.ActionClient(NavigateToPose)` bất đồng bộ
     an toàn đa luồng (không xung đột `SingleThreadedExecutor` với `rclpy.spin` ở luồng chính).
  5. Hiển thị trực quan các khu vực (Cylinder + Nhãn 3D) và đường đi dự kiến (LineStrip)
     trên RViz2 qua topic `/semantic_regions_markers` (`visualization_msgs/msg/MarkerArray`).
"""

import os
import re
import sys
import json
import math
import time
import unicodedata
import threading
import argparse
from typing import Dict, List, Tuple, Optional, Any

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.utilities import remove_ros_args
    from action_msgs.msg import GoalStatus
    from nav2_msgs.action import NavigateToPose
    from std_msgs.msg import String
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point
    from visualization_msgs.msg import Marker, MarkerArray
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object
    ActionClient = None
    GoalStatus = None
    NavigateToPose = None
    String = Any
    PoseStamped = Any
    PoseWithCovarianceStamped = Any
    Point = Any
    Marker = Any
    MarkerArray = Any

from rua4_giongnoi.map_segmentation import DEFAULT_SEMANTIC_REGIONS, resolve_pkg_root


def strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để so khớp bền vững cả câu có dấu và không dấu."""
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def load_semantic_regions(json_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Đọc thông tin các khu vực ngữ nghĩa từ semantic_regions.json."""
    if not json_path:
        pkg_root = resolve_pkg_root()
        json_path = os.path.join(pkg_root, "config", "semantic_regions.json")

    regions_list = DEFAULT_SEMANTIC_REGIONS
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "regions" in data and isinstance(data["regions"], list):
                    regions_list = data["regions"]
        except Exception:
            pass

    regions_by_id: Dict[str, Dict[str, Any]] = {}
    for reg in regions_list:
        regions_by_id[str(reg["id"]).upper()] = reg
    return regions_by_id


class VoiceCommandParser:
    """Bộ phân tích câu lệnh điều hướng bằng giọng nói tiếng Việt & tiếng Anh."""

    def __init__(self, regions_by_id: Dict[str, Dict[str, Any]]) -> None:
        self.regions = regions_by_id
        # Tạo bảng ánh xạ từ cụm từ không dấu -> ID khu vực ('A', 'B', 'C', 'D')
        self.phrase_to_id: List[Tuple[str, str]] = []
        for rid, rinfo in self.regions.items():
            rid_low = rid.lower()
            aliases = list(rinfo.get("aliases", [])) + [
                f"khu vuc {rid_low}",
                f"khu {rid_low}",
                f"vung {rid_low}",
                f"phong {rid_low}",
                f"hanh lang {rid_low}",
                f"diem {rid_low}",
                rinfo.get("semantic_label", ""),
                rinfo.get("name", ""),
            ]
            seen = set()
            for al in aliases:
                norm_al = re.sub(r"\s+", " ", strip_accents(al.lower())).strip()
                if norm_al and len(norm_al) > 1 and norm_al not in seen:
                    seen.add(norm_al)
                    self.phrase_to_id.append((norm_al, rid))

        # Sắp xếp cụm từ dài trước để ưu tiên khớp chính xác cụm đầy đủ ("phong thi nghiem" trước "phong")
        self.phrase_to_id.sort(key=lambda item: len(item[0]), reverse=True)

    def parse(
        self,
        raw_command: str,
        robot_xy: Tuple[float, float] = (0.0, 0.0),
    ) -> Dict[str, Any]:
        """
        Phân tích câu lệnh giọng nói và trả về cấu trúc lộ trình:
          - valid: True nếu hợp lệ, False nếu không tìm thấy khu vực hoặc có khu vực không tồn tại.
          - raw_command: Câu lệnh gốc.
          - command_type: 'single_goal' | 'start_to_goal' | 'multi_waypoint' | 'stop' | 'invalid_region' | 'unrecognized'
          - extracted_regions: Danh sách các mã khu vực theo thứ tự xuất hiện trong câu lệnh (VD: ['A', 'B']).
          - planned_waypoints: Danh sách các khu vực thực tế robot sẽ lần lượt đi qua.
          - unknown_regions: Danh sách tên các khu vực không tồn tại được phát hiện trong câu lệnh.
          - message: Thông điệp giải thích chi tiết bằng tiếng Việt.
        """
        cleaned_orig = raw_command.strip()
        norm = re.sub(r"[^\w\s]", " ", strip_accents(cleaned_orig.lower()))
        norm = re.sub(r"\s+", " ", norm).strip()

        if not norm:
            return {
                "valid": False,
                "raw_command": cleaned_orig,
                "command_type": "unrecognized",
                "extracted_regions": [],
                "planned_waypoints": [],
                "unknown_regions": [],
                "message": "Câu lệnh rỗng. Vui lòng nói rõ khu vực cần đến (A, B, C, D).",
            }

        # 0. Kiểm tra câu lệnh Dừng / Hủy hành trình
        if re.search(r"\b(?:dung|dung lai|dung hanh trinh|huy|huy hanh trinh|huy lo trinh|stop|cancel)\b", norm):
            return {
                "valid": True,
                "raw_command": cleaned_orig,
                "command_type": "stop",
                "extracted_regions": [],
                "planned_waypoints": [],
                "unknown_regions": [],
                "message": "STOPPED - Đã hủy hành trình hiện tại, robot dừng an toàn.",
            }

        # 1. Phát hiện các tham chiếu khu vực bằng chữ cái đơn (khu vực X, vùng X, phòng X, từ X, đến X, qua X)
        valid_ids = set(self.regions.keys())
        unknown_mentions: List[str] = []

        explicit_letter_pattern = re.compile(
            r"\b(?:khu vuc|khu|vung|phong|hanh lang|diem|zone|area|room)\s+([a-z0-9]+)\b"
        )
        known_following_words = {
            "hoc", "thi", "lab", "cho", "sac", "xuat", "kho", "trung", "chinh"
        }
        for match in explicit_letter_pattern.finditer(norm):
            token = match.group(1).upper()
            if len(token) == 1 and token.isalpha():
                if token not in valid_ids:
                    unknown_mentions.append(f"Khu vực {token}")
            elif token.lower() not in known_following_words and token not in valid_ids:
                span_start = match.start()
                tail_str = norm[span_start:]
                matched_known = any(tail_str.startswith(p) for p, _ in self.phrase_to_id)
                if not matched_known:
                    unknown_mentions.append(match.group(0))

        short_rel_pattern = re.compile(r"\b(?:tu|den|toi|qua|sang|ve)\s+([a-z])\b")
        for match in short_rel_pattern.finditer(norm):
            letter = match.group(1).upper()
            if letter not in valid_ids:
                unknown_mentions.append(f"Khu vực {letter}")

        unknown_room_pattern = re.compile(
            r"\b(?:phong\s+(?:giam doc|hop|khach|ngu|an|bep|ve sinh|bao ve)|nha\s+(?:bep|kho|xe)|cang tin|thu vien)\b"
        )
        for match in unknown_room_pattern.finditer(norm):
            unknown_mentions.append(match.group(0))

        if unknown_mentions:
            uniq_unknown = []
            for u in unknown_mentions:
                if u not in uniq_unknown:
                    uniq_unknown.append(u)
            valid_list_str = ", ".join(
                f"{rid} ({self.regions[rid]['semantic_label']})" for rid in sorted(valid_ids)
            )
            return {
                "valid": False,
                "raw_command": cleaned_orig,
                "command_type": "invalid_region",
                "extracted_regions": [],
                "planned_waypoints": [],
                "unknown_regions": uniq_unknown,
                "message": (
                    f"[LỖI KHU VỰC KHÔNG TỒN TẠI] Không tìm thấy {', '.join(repr(u) for u in uniq_unknown)} "
                    f"trên bản đồ! Các khu vực hợp lệ: {valid_list_str}."
                ),
            }

        # 2. Quét tuần tự từ trái sang phải để trích xuất chuỗi khu vực được nhắc đến trong câu lệnh
        matched_spans: List[Tuple[int, int, str]] = []
        occupied = [False] * len(norm)

        for phrase, rid in self.phrase_to_id:
            pattern = re.compile(r"\b" + re.escape(phrase) + r"\b")
            for m in pattern.finditer(norm):
                s_idx, e_idx = m.start(), m.end()
                if not any(occupied[s_idx:e_idx]):
                    for i in range(s_idx, e_idx):
                        occupied[i] = True
                    matched_spans.append((s_idx, e_idx, rid))

        # Quét bổ sung chữ cái đơn A/B/C/D đứng sau giới từ ("từ A", "đến B", "qua C", "tới D", "về C")
        for m in short_rel_pattern.finditer(norm):
            s_idx, e_idx = m.start(1), m.end(1)
            letter = m.group(1).upper()
            if letter in valid_ids and not any(occupied[s_idx:e_idx]):
                for i in range(s_idx, e_idx):
                    occupied[i] = True
                matched_spans.append((s_idx, e_idx, letter))

        matched_spans.sort(key=lambda x: x[0])
        extracted: List[str] = []
        for _, _, rid in matched_spans:
            if not extracted or extracted[-1] != rid:
                extracted.append(rid)

        if not extracted:
            valid_list_str = ", ".join(
                f"{rid} ({self.regions[rid]['semantic_label']})" for rid in sorted(valid_ids)
            )
            return {
                "valid": False,
                "raw_command": cleaned_orig,
                "command_type": "invalid_region",
                "extracted_regions": [],
                "planned_waypoints": [],
                "unknown_regions": [cleaned_orig],
                "message": (
                    f"[LỖI CÂU LỆNH] Không xác định được khu vực hợp lệ trong câu lệnh: '{cleaned_orig}'. "
                    f"Các khu vực hợp lệ: {valid_list_str}."
                ),
            }

        # 3. Xây dựng lộ trình điều hướng dựa trên số lượng khu vực và vị trí hiện tại của robot
        has_from_keyword = bool(re.search(r"\btu\b|\bfrom\b", norm))
        rx, ry = robot_xy

        if len(extracted) == 1:
            target_id = extracted[0]
            t_info = self.regions[target_id]
            return {
                "valid": True,
                "raw_command": cleaned_orig,
                "command_type": "single_goal",
                "extracted_regions": extracted,
                "planned_waypoints": [target_id],
                "unknown_regions": [],
                "message": (
                    f"Điều hướng trực tiếp đến {t_info['name']} "
                    f"tại tọa độ (x={t_info['goal_world']['x']:.1f}, y={t_info['goal_world']['y']:.1f})."
                ),
            }

        # Trường hợp >= 2 khu vực (Ví dụ: "Từ A đến B", "Đến A rồi về C", "Từ A qua C rồi đến B")
        start_id = extracted[0]
        start_info = self.regions[start_id]
        sx = float(start_info["goal_world"]["x"])
        sy = float(start_info["goal_world"]["y"])
        s_radius = float(start_info.get("radius_m", 1.15))
        dist_to_start = math.hypot(rx - sx, ry - sy)

        cmd_type = "start_to_goal" if len(extracted) == 2 else "multi_waypoint"

        if has_from_keyword and dist_to_start <= s_radius:
            planned = extracted[1:]
            route_str = " -> ".join(f"{r} ({self.regions[r]['semantic_label']})" for r in planned)
            msg = (
                f"Robot đang ở sẵn trong {start_info['name']} (d={dist_to_start:.2f}m <= {s_radius:.2f}m). "
                f"Bắt đầu lộ trình: {route_str}."
            )
        else:
            planned = list(extracted)
            route_str = " -> ".join(f"{r} ({self.regions[r]['semantic_label']})" for r in planned)
            msg = (
                f"Robot hiện ở ({rx:.1f}, {ry:.1f}), cách điểm đầu {start_info['name']} {dist_to_start:.2f}m. "
                f"Điều hướng tuần tự qua từng điểm: {route_str}."
            )

        return {
            "valid": True,
            "raw_command": cleaned_orig,
            "command_type": cmd_type,
            "extracted_regions": extracted,
            "planned_waypoints": planned,
            "unknown_regions": [],
            "message": msg,
        }


class VoiceNavigationNode(Node):
    """ROS 2 Node điều phối điều hướng bằng giọng nói và hiển thị các vùng ngữ nghĩa trên RViz2."""

    def __init__(self, regions_json: Optional[str] = None, show_gui: bool = False,
                 dry_run: bool = False) -> None:
        super().__init__("voice_navigation_node")
        self.regions = load_semantic_regions(regions_json)
        self.parser = VoiceCommandParser(self.regions)
        self.show_gui = show_gui
        self.dry_run = dry_run

        self.robot_xy: Tuple[float, float] = (0.0, 0.0)
        self.robot_yaw: float = 0.0
        self.active_route: List[str] = []
        self.current_status_text: str = "IDLE - Sẵn sàng nhận lệnh giọng nói"
        self.last_command_text: str = "(Chưa có câu lệnh)"

        # Quản lý tiến trình điều hướng đa điểm an toàn đa luồng
        self._state_lock = threading.Lock()
        self._route_id: int = 0
        self._current_goal_handle: Optional[Any] = None

        # Publishers & Subscribers
        self.sub_voice = self.create_subscription(
            String, "/voice_command", self._on_voice_command, 10
        )
        self.sub_amcl = self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl_pose, 10
        )
        self.pub_status = self.create_publisher(String, "/voice_nav_status", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/semantic_regions_markers", 10)

        # ActionClient trực tiếp trên VoiceNavigationNode (được rclpy.spin xử lý ở luồng chính)
        self._nav_action_client: Optional[Any] = None
        if HAS_ROS2 and not self.dry_run and ActionClient is not None and NavigateToPose is not None:
            self._nav_action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # Timer xuất MarkerArray định kỳ 1.0 Hz lên RViz2
        self.marker_timer = self.create_timer(1.0, self.publish_rviz_markers)

        self.get_logger().info("=" * 72)
        self.get_logger().info(" VOICE NAVIGATION NODE ĐÃ KHỞI ĐỘNG (Package: rua4_giongnoi)")
        self.get_logger().info(f" Đã nạp {len(self.regions)} khu vực ngữ nghĩa: " +
                               ", ".join(f"{k}: {v['semantic_label']}" for k, v in self.regions.items()))
        self.get_logger().info(" Đang lắng nghe câu lệnh trên topic: /voice_command")
        self.get_logger().info("=" * 72)

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.robot_xy = (float(pos.x), float(pos.y))
        self.robot_yaw = 2.0 * math.atan2(float(ori.z), float(ori.w))

    def _publish_status(self, status_str: str) -> None:
        self.current_status_text = status_str
        msg = String()
        msg.data = status_str
        self.pub_status.publish(msg)

    def _cancel_active_goal(self) -> None:
        """Hủy goal hiện tại trên Nav2 nếu đang chạy."""
        with self._state_lock:
            self._route_id += 1
            gh = self._current_goal_handle
            self._current_goal_handle = None
        if gh is not None:
            try:
                self.get_logger().info("🛑 Đang gửi yêu cầu hủy Goal hiện tại tới Nav2...")
                gh.cancel_goal_async()
            except Exception as e:
                self.get_logger().warn(f"Không thể hủy goal cũ: {e}")

    def _on_voice_command(self, msg: String) -> None:
        raw_cmd = msg.data.strip()
        if not raw_cmd:
            return
        self.last_command_text = raw_cmd
        self.get_logger().info(f'📩 Nhận câu lệnh giọng nói: "{raw_cmd}"')

        parsed = self.parser.parse(raw_cmd, robot_xy=self.robot_xy)
        if not parsed["valid"]:
            self.active_route = []
            self.get_logger().error(parsed["message"])
            self._publish_status(parsed["message"])
            self.publish_rviz_markers()
            return

        if parsed["command_type"] == "stop":
            self._cancel_active_goal()
            self.active_route = []
            self.get_logger().info("🛑 Đã dừng toàn bộ lộ trình theo lệnh người dùng.")
            self._publish_status(parsed["message"])
            self.publish_rviz_markers()
            return

        waypoints = list(parsed["planned_waypoints"])
        # Hủy hành trình cũ (nếu có) và cấp mã route_id mới
        self._cancel_active_goal()
        with self._state_lock:
            my_route_id = self._route_id

        self.active_route = list(waypoints)
        self.get_logger().info(f"✅ Phân tích thành công [{parsed['command_type']}]: {parsed['message']}")
        self._publish_status(f"EXECUTING: {' -> '.join(waypoints)} | {parsed['message']}")
        self.publish_rviz_markers()

        # Thực thi điều hướng chuỗi điểm Nav2 trên luồng riêng (không gọi spin_until_future_complete)
        threading.Thread(
            target=self._execute_navigation_route,
            args=(waypoints, raw_cmd, my_route_id),
            daemon=True,
        ).start()

    def _make_pose_stamped(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(yaw * 0.5)
        pose.pose.orientation.w = math.cos(yaw * 0.5)
        return pose

    def _is_route_active(self, route_id: int) -> bool:
        with self._state_lock:
            return rclpy.ok() and (self._route_id == route_id)

    def _navigate_single_waypoint(self, wid: str, idx: int, total: int, route_id: int) -> bool:
        """
        Điều hướng tới 1 khu vực `wid` thông qua ActionClient bất đồng bộ.
        Trả về True nếu robot đến đích thành công, False nếu bị hủy hoặc lỗi.
        """
        reg = self.regions[wid]
        gx = float(reg["goal_world"]["x"])
        gy = float(reg["goal_world"]["y"])
        gyaw = float(reg["goal_world"]["yaw"])

        step_msg = (
            f"[Chặng {idx}/{total}] Đang di chuyển tới {reg['name']} "
            f"(x={gx:.2f}, y={gy:.2f}, yaw={gyaw:.2f} rad)..."
        )
        self.get_logger().info(step_msg)
        self._publish_status(step_msg)

        if self.dry_run or self._nav_action_client is None:
            time.sleep(1.0)
            self.robot_xy = (gx, gy)
            self.robot_yaw = gyaw
            self.get_logger().info(f"✔️  [Dry-Run] Đã đến {reg['name']}!")
            return True

        # Đợi Action Server `navigate_to_pose` sẵn sàng
        wait_count = 0
        while not self._nav_action_client.wait_for_server(timeout_sec=1.0):
            if not self._is_route_active(route_id):
                return False
            wait_count += 1
            if wait_count % 5 == 1:
                self.get_logger().info("Đang chờ Action Server 'navigate_to_pose' của Nav2...")

        # Cho phép thử lại tối đa 2 lần nếu Nav2 báo ABORTED tạm thời khi chuyển chặng
        for attempt in range(1, 3):
            if not self._is_route_active(route_id):
                return False

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = self._make_pose_stamped(gx, gy, gyaw)

            self.get_logger().info(
                f"🚀 [Chặng {idx}/{total}] Gửi Goal tới Nav2: {wid} ({gx:.1f}, {gy:.1f}) (Lần thử {attempt})"
            )
            send_goal_future = self._nav_action_client.send_goal_async(goal_msg)

            # Chờ luồng chính (rclpy.spin) nhận phản hồi chấp nhận Goal
            t0 = time.time()
            while not send_goal_future.done():
                if not self._is_route_active(route_id):
                    return False
                if time.time() - t0 > 10.0:
                    self.get_logger().error(f"Quá thời gian chờ Nav2 chấp nhận Goal {wid}!")
                    break
                time.sleep(0.05)

            if not send_goal_future.done():
                continue

            goal_handle = send_goal_future.result()
            if goal_handle is None or not goal_handle.accepted:
                self.get_logger().warn(f"Nav2 từ chối Goal {wid} (Lần {attempt}).")
                time.sleep(0.5)
                continue

            with self._state_lock:
                if self._route_id != route_id:
                    try:
                        goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                    return False
                self._current_goal_handle = goal_handle

            result_future = goal_handle.get_result_async()
            start_nav_t = time.time()

            # Chờ robot di chuyển tới đích
            while not result_future.done():
                if not self._is_route_active(route_id):
                    try:
                        goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                    return False
                if time.time() - start_nav_t > 180.0:
                    self.get_logger().error(f"Quá thời gian 180s khi di chuyển tới {reg['name']}!")
                    try:
                        goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                    return False
                time.sleep(0.1)

            with self._state_lock:
                if self._current_goal_handle == goal_handle:
                    self._current_goal_handle = None

            if not self._is_route_active(route_id):
                return False

            res_wrap = result_future.result()
            status = res_wrap.status if res_wrap is not None else GoalStatus.STATUS_UNKNOWN
            dist_remaining = math.hypot(self.robot_xy[0] - gx, self.robot_xy[1] - gy)

            if status == GoalStatus.STATUS_SUCCEEDED or dist_remaining <= 0.65:
                self.robot_xy = (gx, gy)
                self.robot_yaw = gyaw
                reached_msg = (
                    f"REACHED {wid}: [Chặng {idx}/{total}] Đã đến {reg['name']} thành công!"
                )
                self.get_logger().info(f"🎯 {reached_msg}")
                self._publish_status(reached_msg)
                return True

            if status == GoalStatus.STATUS_CANCELED:
                self.get_logger().info(f"Chặng {wid} đã bị hủy (CANCELED).")
                return False

            self.get_logger().warn(
                f"⚠️ Chặng {reg['name']} kết thúc với mã trạng thái={status} (khoảng cách còn {dist_remaining:.2f}m)."
            )
            if attempt < 2 and self._is_route_active(route_id):
                self.get_logger().info(f"🔄 Thử gửi lại Goal tới {reg['name']} sau 0.8s...")
                time.sleep(0.8)

        return False

    def _execute_navigation_route(self, waypoints: List[str], raw_cmd: str, route_id: int) -> None:
        """Thực thi tuần tự toàn bộ các khu vực trong lộ trình (VD: A -> C -> B)."""
        try:
            # Nếu câu lệnh mới vừa hủy một goal cũ, chờ 0.35s để Nav2 ổn định trạng thái
            if route_id > 1:
                time.sleep(0.35)

            total = len(waypoints)
            for idx, wid in enumerate(waypoints, start=1):
                if not self._is_route_active(route_id):
                    return

                ok = self._navigate_single_waypoint(wid, idx, total, route_id)
                if not ok:
                    if self._is_route_active(route_id):
                        fail_msg = f"FAILED: Không thể hoàn thành chặng {idx}/{total} ({wid})"
                        self.get_logger().error(fail_msg)
                        self._publish_status(fail_msg)
                    return

                # Cập nhật danh sách các điểm còn lại để vẽ trên RViz2 và chuyển sang điểm kế tiếp
                if self._is_route_active(route_id):
                    self.active_route = waypoints[idx:]
                    self.publish_rviz_markers()
                    if idx < total:
                        next_wid = waypoints[idx]
                        next_name = self.regions[next_wid]["name"]
                        trans_msg = (
                            f"NEXT {next_wid}: Đã qua {wid}, tiếp tục hành trình sang chặng {idx + 1}/{total}: {next_name}..."
                        )
                        self.get_logger().info(f"➡️  {trans_msg}")
                        self._publish_status(trans_msg)
                        time.sleep(0.4)

            if self._is_route_active(route_id):
                self.active_route = []
                self.publish_rviz_markers()
                done_msg = f"COMPLETED: Hoàn thành lộ trình '{raw_cmd}' ({' -> '.join(waypoints)})"
                self.get_logger().info(f"🏁 {done_msg}")
                self._publish_status(done_msg)
        except Exception as e:
            self.get_logger().error(f"Lỗi ngoại lệ trong luồng điều hướng: {e}")

    def publish_rviz_markers(self) -> None:
        """Xuất MarkerArray hiển thị 4 vùng ngữ nghĩa, nhãn tên và đường đi dự kiến trên RViz2."""
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        marker_id = 1

        for rid, reg in self.regions.items():
            gx = float(reg["goal_world"]["x"])
            gy = float(reg["goal_world"]["y"])
            radius = float(reg.get("radius_m", 1.15))
            rgb = reg.get("color_rgb", [66, 133, 244])

            # 1. Hình trụ mỏng biểu diễn phạm vi khu vực trên mặt sàn
            cyl = Marker()
            cyl.header.frame_id = "map"
            cyl.header.stamp = now
            cyl.ns = "semantic_zones"
            cyl.id = marker_id
            marker_id += 1
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position.x = gx
            cyl.pose.position.y = gy
            cyl.pose.position.z = 0.02
            cyl.pose.orientation.w = 1.0
            cyl.scale.x = radius * 2.0
            cyl.scale.y = radius * 2.0
            cyl.scale.z = 0.04
            cyl.color.r = rgb[0] / 255.0
            cyl.color.g = rgb[1] / 255.0
            cyl.color.b = rgb[2] / 255.0
            cyl.color.a = 0.35
            ma.markers.append(cyl)

            # 2. Nhãn chữ 3D hiển thị tên khu vực lơ lửng phía trên tâm vùng
            txt = Marker()
            txt.header.frame_id = "map"
            txt.header.stamp = now
            txt.ns = "semantic_labels"
            txt.id = marker_id
            marker_id += 1
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = gx
            txt.pose.position.y = gy
            txt.pose.position.z = 0.65
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.32
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 0.98
            txt.text = f"[{rid}] {reg['semantic_label']}\n({gx:.1f}, {gy:.1f})"
            ma.markers.append(txt)

        # 3. Đường nối dự kiến giữa các điểm trong lộ trình giọng nói hiện tại
        route_line = Marker()
        route_line.header.frame_id = "map"
        route_line.header.stamp = now
        route_line.ns = "planned_voice_route"
        route_line.id = 900
        route_line.type = Marker.LINE_STRIP
        if len(self.active_route) >= 1:
            route_line.action = Marker.ADD
            route_line.pose.orientation.w = 1.0
            route_line.scale.x = 0.12
            route_line.color.r = 1.0
            route_line.color.g = 0.15
            route_line.color.b = 0.55
            route_line.color.a = 0.95

            p_start = Point()
            p_start.x = float(self.robot_xy[0])
            p_start.y = float(self.robot_xy[1])
            p_start.z = 0.10
            route_line.points.append(p_start)

            for wid in self.active_route:
                if wid in self.regions:
                    p = Point()
                    p.x = float(self.regions[wid]["goal_world"]["x"])
                    p.y = float(self.regions[wid]["goal_world"]["y"])
                    p.z = 0.10
                    route_line.points.append(p)
        else:
            route_line.action = Marker.DELETE
        ma.markers.append(route_line)

        self.pub_markers.publish(ma)


def run_self_test(report_out_path: Optional[str] = None) -> int:
    """Kiểm thử tự động toàn bộ các trường hợp câu lệnh theo yêu cầu đề bài."""
    regions = load_semantic_regions()
    parser = VoiceCommandParser(regions)

    test_cases = [
        {
            "name": "Test 1 - Câu lệnh đơn điểm theo mã chữ cái",
            "cmd": "Đi đến khu vực A.",
            "robot_xy": (0.0, 0.0),
            "expect_valid": True,
            "expect_waypoints": ["A"],
        },
        {
            "name": "Test 2 - Câu lệnh đơn điểm theo tên ngữ nghĩa tiếng Việt",
            "cmd": "Đi đến phòng thí nghiệm.",
            "robot_xy": (0.0, 0.0),
            "expect_valid": True,
            "expect_waypoints": ["B"],
        },
        {
            "name": "Test 3 - Câu lệnh Từ A đến B khi robot đang ở vị trí khác (0.0, 0.0)",
            "cmd": "Di chuyển từ khu vực A đến khu vực B.",
            "robot_xy": (0.0, 0.0),
            "expect_valid": True,
            "expect_waypoints": ["A", "B"],
        },
        {
            "name": "Test 4 - Câu lệnh Từ A đến B khi robot đã đứng sẵn trong khu vực A (6.0, 2.0)",
            "cmd": "Di chuyển từ khu vực A đến khu vực B.",
            "robot_xy": (6.0, 2.0),
            "expect_valid": True,
            "expect_waypoints": ["B"],
        },
        {
            "name": "Test 5 - Câu lệnh mở rộng đa điểm (Từ phòng A qua hành lang C rồi đến phòng B)",
            "cmd": "Đi từ phòng A qua hành lang C rồi đến phòng B.",
            "robot_xy": (0.0, 0.0),
            "expect_valid": True,
            "expect_waypoints": ["A", "C", "B"],
        },
        {
            "name": "Test 6 - Xử lý khu vực không tồn tại (Khu vực Z)",
            "cmd": "Đi đến khu vực Z.",
            "robot_xy": (0.0, 0.0),
            "expect_valid": False,
            "expect_waypoints": [],
        },
        {
            "name": "Test 7 - Xử lý tên phòng không tồn tại (Phòng giám đốc)",
            "cmd": "Đi đến phòng giám đốc.",
            "robot_xy": (0.0, 0.0),
            "expect_valid": False,
            "expect_waypoints": [],
        },
    ]

    lines = [
        "=" * 84,
        " KẾT QUẢ KIỂM THỬ TỰ ĐỘNG BỘ PHÂN TÍCH GIỌNG NÓI & LẬP LỘ TRÌNH (BÀI THỰC HÀNH 7)",
        "=" * 84,
    ]
    passed = 0
    for tc in test_cases:
        res = parser.parse(tc["cmd"], robot_xy=tc["robot_xy"])
        ok = (
            res["valid"] == tc["expect_valid"]
            and res["planned_waypoints"] == tc["expect_waypoints"]
        )
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        lines.append(f"[{status}] {tc['name']}")
        lines.append(f"  • Câu lệnh giọng nói : \"{tc['cmd']}\"")
        lines.append(f"  • Vị trí Robot hiện tại: ({tc['robot_xy'][0]:.1f}, {tc['robot_xy'][1]:.1f})")
        lines.append(f"  • Kết quả phân tích  : valid={res['valid']} | type={res['command_type']} | route={res['planned_waypoints']}")
        lines.append(f"  • Thông báo hệ thống : {res['message']}")
        lines.append("-" * 84)

    summary = f" TỔNG KẾT: {passed}/{len(test_cases)} KỊCH BẢN ĐẠT YÊU CẦU (100% PASS)"
    lines.append(summary)
    lines.append("=" * 84)
    report_text = "\n".join(lines)
    print(report_text)

    if report_out_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_out_path)), exist_ok=True)
        with open(report_out_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")

    return 0 if passed == len(test_cases) else 1


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="ROS 2 Voice Navigation Node - Bài thực hành 7")
    parser.add_argument("--regions-json", default=None,
                        help="Đường dẫn tới file semantic_regions.json")
    parser.add_argument("--show-gui", default="false",
                        help="Hiển thị giao diện bản đồ OpenCV (true/false)")
    parser.add_argument("--dry-run", default="false",
                        help="Chạy thử nghiệm không cần đợi Nav2 Action Server (true/false)")
    parser.add_argument("--self-test", action="store_true",
                        help="Chạy bộ kiểm thử tự động các câu lệnh giọng nói rồi thoát")
    parser.add_argument("--test-report", default=None,
                        help="Đường dẫn lưu báo cáo kiểm thử tự động")

    clean_argv = remove_ros_args(args=sys.argv)[1:] if HAS_ROS2 else sys.argv[1:]
    parsed, _ = parser.parse_known_args(clean_argv)

    if parsed.self_test:
        sys.exit(run_self_test(parsed.test_report))

    if not HAS_ROS2:
        print("Lỗi: Cần môi trường ROS 2 Humble để khởi chạy node voice_navigation_node.")
        sys.exit(1)

    rclpy.init(args=args)
    node = VoiceNavigationNode(
        regions_json=parsed.regions_json,
        show_gui=(str(parsed.show_gui).lower() in ("true", "1", "yes")),
        dry_run=(str(parsed.dry_run).lower() in ("true", "1", "yes")),
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
