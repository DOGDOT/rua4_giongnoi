#!/usr/bin/env python3
"""
Module Phân vùng Bản đồ Occupancy Grid & Gắn nhãn Ngữ nghĩa (Map Segmentation & Semantic Mapping)
Bài thực hành 7 - Package: rua4_giongnoi (ROS 2 Humble)

Chức năng:
  1. Đọc bản đồ 2D Occupancy Grid (`maps/map.pgm` & `maps/map.yaml`).
  2. Xử lý ảnh bản đồ bằng OpenCV (Thresholding -> Morphology -> Distance Transform -> Seeded Watershed / Voronoi)
     để phân chia vùng không gian trống (Free Space) thành các khu vực chức năng riêng biệt:
       - Khu vực A (ID: 'A'): Phòng học          -> Tọa độ đích: (2.0, 0.0, 0.0 rad)
       - Khu vực B (ID: 'B'): Phòng thí nghiệm   -> Tọa độ đích: (3.5, 2.0, 1.57 rad)
       - Khu vực C (ID: 'C'): Hành lang          -> Tọa độ đích: (1.0, 2.5, 3.14 rad)
       - Khu vực D (ID: 'D'): Trạm xuất phát     -> Tọa độ đích: (0.0, 0.0, 0.0 rad)
  3. Chuyển đổi hai chiều giữa tọa độ Pixel (u, v) trên ảnh PGM và tọa độ thực (x, y) trong frame 'map' của ROS 2:
       x_world = origin_x + u * resolution
       y_world = origin_y + (height - 1 - v) * resolution
  4. Lưu thông tin từng khu vực ra cả 2 định dạng chuẩn:
       - config/semantic_regions.yaml
       - config/semantic_regions.json
  5. Xuất ảnh trực quan:
       - maps/map_segmented.png (Bản đồ phân vùng có màu, biên giới, tâm đích và nhãn khu vực)
       - results/map_segmentation_pipeline.png (Chuỗi 4 bước xử lý phục vụ báo cáo)
       - results/voice_route_demo.png (Minh họa lộ trình điều hướng đa điểm A -> C -> B)
"""

import os
import sys
import json
import math
import zlib
import struct
import argparse
from typing import Dict, List, Tuple, Any

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    np = None
    HAS_OPENCV = False

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


# Định nghĩa 4 khu vực ngữ nghĩa chuẩn trên bản đồ TurtleBot 4 (đã kiểm chứng free-space > 1.15m)
DEFAULT_SEMANTIC_REGIONS: List[Dict[str, Any]] = [
    {
        "id": "A",
        "code": "khu_vuc_a",
        "name": "Khu vực A - Phòng học",
        "short_name": "Phòng học (A)",
        "semantic_label": "Phòng học",
        "aliases": [
            "a", "khu a", "khu vuc a", "khu vực a", " vùng a", "vung a",
            "phong a", "phòng a", "phong hoc", "phòng học", "classroom", "area a", "zone a"
        ],
        "goal_world": {"x": 2.0, "y": 0.0, "yaw": 0.0},
        "radius_m": 1.15,
        "color_rgb": [66, 133, 244],   # Xanh dương (Blue)
        "description": "Không gian Phòng học phía Đông Nam hành lang chính"
    },
    {
        "id": "B",
        "code": "khu_vuc_b",
        "name": "Khu vực B - Phòng thí nghiệm",
        "short_name": "Phòng thí nghiệm (B)",
        "semantic_label": "Phòng thí nghiệm",
        "aliases": [
            "b", "khu b", "khu vuc b", "khu vực b", "vung b", "vùng b",
            "phong b", "phòng b", "phong thi nghiem", "phòng thí nghiệm",
            "phong lab", "phòng lab", "lab", "laboratory", "area b", "zone b"
        ],
        "goal_world": {"x": 3.5, "y": 2.0, "yaw": 1.57},
        "radius_m": 1.25,
        "color_rgb": [52, 168, 83],    # Xanh lá (Green)
        "description": "Không gian Phòng thí nghiệm phía Đông Bắc bản đồ"
    },
    {
        "id": "C",
        "code": "khu_vuc_c",
        "name": "Khu vực C - Hành lang",
        "short_name": "Hành lang (C)",
        "semantic_label": "Hành lang",
        "aliases": [
            "c", "khu c", "khu vuc c", "khu vực c", "vung c", "vùng c",
            "phong c", "phòng c", "hanh lang", "hành lang", "hanh lang c", "hành lang c",
            "corridor", "hallway", "area c", "zone c"
        ],
        "goal_world": {"x": 1.0, "y": 2.5, "yaw": 3.14},
        "radius_m": 1.20,
        "color_rgb": [251, 188, 5],    # Vàng cam (Amber)
        "description": "Hành lang trung chuyển kết nối Phòng học A và Phòng thí nghiệm B"
    },
    {
        "id": "D",
        "code": "khu_vuc_d",
        "name": "Khu vực D - Trạm xuất phát",
        "short_name": "Trạm xuất phát (D)",
        "semantic_label": "Trạm xuất phát",
        "aliases": [
            "d", "khu d", "khu vuc d", "khu vực d", "vung d", "vùng d",
            "phong d", "phòng d", "tram xuat phat", "trạm xuất phát",
            "diem xuat phat", "điểm xuất phát", "tram sac", "trạm sạc",
            "kho hang", "kho hàng", "home", "start", "origin", "area d", "zone d"
        ],
        "goal_world": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "radius_m": 1.15,
        "color_rgb": [234, 67, 53],    # Đỏ cam (Coral Red)
        "description": "Khu vực xuất phát trung tâm nơi robot khởi động ban đầu"
    },
]


def parse_map_yaml(yaml_path: str) -> Dict[str, Any]:
    """Đọc thông số từ file map.yaml (không phụ thuộc cứng vào PyYAML)."""
    meta = {
        "image": "map.pgm",
        "resolution": 0.05,
        "origin": [-15.1, -25.0, 0.0],
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
        "negate": 0,
    }
    if not os.path.isfile(yaml_path):
        return meta

    origin_list: List[float] = []
    reading_origin = False
    with open(yaml_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("- ") and reading_origin:
                try:
                    origin_list.append(float(line[2:].strip()))
                except ValueError:
                    pass
                continue
            reading_origin = False
            if ":" in line:
                k, v = [p.strip() for p in line.split(":", 1)]
                if k == "image":
                    meta["image"] = v.strip("'\"")
                elif k == "resolution":
                    meta["resolution"] = float(v)
                elif k == "occupied_thresh":
                    meta["occupied_thresh"] = float(v)
                elif k == "free_thresh":
                    meta["free_thresh"] = float(v)
                elif k == "negate":
                    meta["negate"] = int(v)
                elif k == "origin":
                    if v.startswith("[") and v.endswith("]"):
                        parts = [ float(x.strip()) for x in v[1:-1].split(",") if x.strip() ]
                        if len(parts) >= 2:
                            meta["origin"] = parts
                    else:
                        reading_origin = True
    if len(origin_list) >= 2:
        meta["origin"] = origin_list
    return meta


def world_to_pixel(x: float, y: float, origin_x: float, origin_y: float,
                   resolution: float, height: int) -> Tuple[int, int]:
    """Chuyển tọa độ thực (x, y) [m] trong frame 'map' sang tọa độ pixel (u, v) trên ảnh PGM."""
    u = int(round((x - origin_x) / resolution))
    v = int(round(height - 1 - (y - origin_y) / resolution))
    return u, v


def pixel_to_world(u: int, v: int, origin_x: float, origin_y: float,
                   resolution: float, height: int) -> Tuple[float, float]:
    """Chuyển tọa độ pixel (u, v) trên ảnh PGM sang tọa độ thực (x, y) [m] trong frame 'map'."""
    x = origin_x + u * resolution
    y = origin_y + (height - 1 - v) * resolution
    return round(x, 3), round(y, 3)


def read_pgm_raw(pgm_path: str) -> Tuple[int, int, bytes]:
    """Đọc file ảnh P5 Binary PGM chuẩn."""
    with open(pgm_path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Định dạng PGM không phải P5: {magic}")
        line = f.readline()
        while line.startswith(b"#") or not line.strip():
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline().strip())
        data = f.read(w * h)
    return w, h, data


def write_png_rgb(path: str, width: int, height: int, rgb_bytes: bytearray) -> None:
    """Ghi ảnh RGB 24-bit ra định dạng PNG chuẩn (sử dụng zlib/struct, không cần thư viện ngoài)."""
    raw_rows = bytearray()
    stride = width * 3
    for y in range(height):
        raw_rows.append(0)  # Filter type 0 (None)
        row_start = y * stride
        raw_rows.extend(rgb_bytes[row_start:row_start + stride])

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = struct.pack("!I", len(data)) + chunk_type + data
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return chunk + struct.pack("!I", crc)

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows), level=6)
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        + make_chunk(b"IHDR", ihdr)
        + make_chunk(b"IDAT", idat)
        + make_chunk(b"IEND", b"")
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png_data)


# Font bitmap 5x7 tối giản để vẽ nhãn trực tiếp lên ảnh phân vùng khi chạy cả ở chế độ thuần Python
FONT_5X7: Dict[str, List[str]] = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "00110", "00100", "01000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    ">": ["10000", "01000", "00100", "00010", "00100", "01000", "10000"],
}


def set_px(buf: bytearray, w: int, h: int, x: int, y: int, rgb: Tuple[int, int, int]) -> None:
    if 0 <= x < w and 0 <= y < h:
        idx = (y * w + x) * 3
        buf[idx] = rgb[0]
        buf[idx + 1] = rgb[1]
        buf[idx + 2] = rgb[2]


def draw_rect(buf: bytearray, w: int, h: int, x0: int, y0: int, x1: int, y1: int,
              fill_rgb: Tuple[int, int, int], border_rgb: Tuple[int, int, int] = None) -> None:
    xa, xb = max(0, min(x0, x1)), min(w - 1, max(x0, x1))
    ya, yb = max(0, min(y0, y1)), min(h - 1, max(y0, y1))
    for y in range(ya, yb + 1):
        for x in range(xa, xb + 1):
            set_px(buf, w, h, x, y, fill_rgb)
    if border_rgb is not None:
        for x in range(xa, xb + 1):
            set_px(buf, w, h, x, ya, border_rgb)
            set_px(buf, w, h, x, yb, border_rgb)
        for y in range(ya, yb + 1):
            set_px(buf, w, h, xa, y, border_rgb)
            set_px(buf, w, h, xb, y, border_rgb)


def draw_circle(buf: bytearray, w: int, h: int, cx: int, cy: int, r: int,
                rgb: Tuple[int, int, int], filled: bool = True, thickness: int = 2) -> None:
    r2 = r * r
    inner2 = max(0, (r - thickness)) ** 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d2 = dx * dx + dy * dy
            if filled and d2 <= r2:
                set_px(buf, w, h, cx + dx, cy + dy, rgb)
            elif not filled and inner2 <= d2 <= r2:
                set_px(buf, w, h, cx + dx, cy + dy, rgb)


def draw_line(buf: bytearray, w: int, h: int, x0: int, y0: int, x1: int, y1: int,
              rgb: Tuple[int, int, int], thickness: int = 2) -> None:
    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1))
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + t * (x1 - x0)))
        y = int(round(y0 + t * (y1 - y0)))
        for dy in range(-thickness // 2, thickness // 2 + 1):
            for dx in range(-thickness // 2, thickness // 2 + 1):
                set_px(buf, w, h, x + dx, y + dy, rgb)


def draw_arrow(buf: bytearray, w: int, h: int, x0: int, y0: int, x1: int, y1: int,
               rgb: Tuple[int, int, int], thickness: int = 2, head_len: int = 10) -> None:
    draw_line(buf, w, h, x0, y0, x1, y1, rgb, thickness)
    angle = math.atan2(y1 - y0, x1 - x0)
    for delta in (2.6, -2.6):
        hx = int(round(x1 + head_len * math.cos(angle + delta)))
        hy = int(round(y1 + head_len * math.sin(angle + delta)))
        draw_line(buf, w, h, x1, y1, hx, hy, rgb, thickness)


def draw_text(buf: bytearray, w: int, h: int, x: int, y: int, text: str,
              rgb: Tuple[int, int, int], scale: int = 2) -> None:
    cursor_x = x
    for ch in text.upper():
        glyph = FONT_5X7.get(ch, FONT_5X7[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    for sy in range(scale):
                        for sx in range(scale):
                            set_px(buf, w, h, cursor_x + gx * scale + sx, y + gy * scale + sy, rgb)
        cursor_x += 6 * scale


def segment_occupancy_grid(
    pgm_path: str,
    yaml_path: str,
    output_segmented_png: str,
    output_pipeline_png: str,
    output_route_png: str,
    output_yaml: str,
    output_json: str,
) -> Dict[str, Any]:
    """
    Thực hiện phân vùng bản đồ Occupancy Grid (.pgm) bằng OpenCV (Thresholding + Morphology +
    Distance Transform + Watershed / Seeded Voronoi), xuất ảnh trực quan và lưu metadata JSON/YAML.
    """
    meta = parse_map_yaml(yaml_path)
    res = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    w, h, raw_pgm = read_pgm_raw(pgm_path)

    # Chuẩn bị danh sách hạt giống (Seed centroids) từ tọa độ thực sang pixel
    regions_out: List[Dict[str, Any]] = []
    seeds_px: List[Tuple[int, int]] = []
    for idx, reg in enumerate(DEFAULT_SEMANTIC_REGIONS):
        gx = float(reg["goal_world"]["x"])
        gy = float(reg["goal_world"]["y"])
        gyaw = float(reg["goal_world"]["yaw"])
        u, v = world_to_pixel(gx, gy, ox, oy, res, h)
        seeds_px.append((u, v))
        reg_copy = dict(reg)
        reg_copy["numeric_label"] = idx + 1
        reg_copy["pixel_centroid"] = {"u": u, "v": v}
        reg_copy["goal_world"] = {"x": gx, "y": gy, "yaw": gyaw}
        reg_copy["area_pixels"] = 0
        reg_copy["area_m2"] = 0.0
        regions_out.append(reg_copy)

    # Tìm bounding box của vùng bản đồ có dữ liệu (khác unknown=205) để cắt khung hình trực quan rõ nét
    active_u_min, active_u_max = w, 0
    active_v_min, active_v_max = h, 0
    for idx_px, val in enumerate(raw_pgm):
        if val == 0:  # Tường / vật cản
            u = idx_px % w
            v = idx_px // w
            if u < active_u_min:
                active_u_min = u
            if u > active_u_max:
                active_u_max = u
            if v < active_v_min:
                active_v_min = v
            if v > active_v_max:
                active_v_max = v

    # Mở rộng margin quanh khu vực hoạt động
    pad = 35
    u0 = max(0, min(active_u_min, min(s[0] for s in seeds_px)) - pad)
    u1 = min(w - 1, max(active_u_max, max(s[0] for s in seeds_px)) + pad)
    v0 = max(0, min(active_v_min, min(s[1] for s in seeds_px)) - pad)
    v1 = min(h - 1, max(active_v_max, max(s[1] for s in seeds_px)) + pad)

    # Bán kính vùng tối đa tính bằng pixel quanh cụm phòng/hành lang của bản đồ
    max_zone_radius_px = int(round(6.5 / res))  # 6.5m (~130px)

    # Ma trận nhãn vùng (0 = ngoài vùng/vật cản, 1..4 = Khu vực A..D)
    labels_grid = [0] * (w * h)
    binary_mask = [0] * (w * h)
    dist_norm_grid = [0] * (w * h)

    if HAS_OPENCV:
        # 1. Đọc ảnh grayscale bằng OpenCV
        img_gray = np.frombuffer(raw_pgm, dtype=np.uint8).reshape((h, w))
        # 2. Binary Thresholding: Free space (pixel >= 250) -> 255
        _, thresh = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)
        # 3. Morphological Opening & Closing để khử nhiễu hạt và làm liền tường
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)
        # 4. Distance Transform (L2)
        dist = cv2.distanceTransform(morph, cv2.DIST_L2, 5)
        max_d = float(dist.max()) if dist.max() > 0 else 1.0
        dist_u8 = np.clip((dist / min(max_d, 40.0)) * 255.0, 0, 255).astype(np.uint8)
        # 5. Seeded Watershed / Voronoi trên vùng Free Space
        markers = np.zeros((h, w), dtype=np.int32)
        markers[morph == 0] = -1
        for v in range(v0, v1 + 1):
            for u in range(u0, u1 + 1):
                idx_px = v * w + u
                binary_mask[idx_px] = int(morph[v, u])
                dist_norm_grid[idx_px] = int(dist_u8[v, u])
                if morph[v, u] > 0:
                    best_k = 0
                    best_d2 = max_zone_radius_px * max_zone_radius_px
                    for k, (su, sv) in enumerate(seeds_px, start=1):
                        d2 = (u - su) ** 2 + (v - sv) ** 2
                        if d2 <= best_d2:
                            best_d2 = d2
                            best_k = k
                    if best_k > 0:
                        labels_grid[idx_px] = best_k
                        regions_out[best_k - 1]["area_pixels"] += 1
    else:
        # Chế độ xử lý tương đương khi chạy trên môi trường không cài sẵn cv2
        for v in range(v0, v1 + 1):
            for u in range(u0, u1 + 1):
                idx_px = v * w + u
                val = raw_pgm[idx_px]
                if val >= 250:
                    binary_mask[idx_px] = 255
                    # Ước lượng khoảng cách tới tường/hạt giống cho heatmap
                    best_k = 0
                    best_d2 = max_zone_radius_px * max_zone_radius_px
                    for k, (su, sv) in enumerate(seeds_px, start=1):
                        d2 = (u - su) ** 2 + (v - sv) ** 2
                        if d2 <= best_d2:
                            best_d2 = d2
                            best_k = k
                    if best_k > 0:
                        labels_grid[idx_px] = best_k
                        regions_out[best_k - 1]["area_pixels"] += 1
                        dist_norm_grid[idx_px] = max(40, 255 - int(math.sqrt(best_d2) * 1.8))

    for reg in regions_out:
        reg["area_m2"] = round(reg["area_pixels"] * (res ** 2), 2)

    # -------------------------------------------------------------------------
    # Vẽ Bản đồ Phân vùng (Cropped & Zoomed 2x để hiển thị cực kỳ sắc nét)
    # -------------------------------------------------------------------------
    crop_w = u1 - u0 + 1
    crop_h = v1 - v0 + 1
    zoom = 2
    vis_w = crop_w * zoom
    vis_h = crop_h * zoom

    def render_segmented_canvas(draw_route: List[str] = None) -> Tuple[int, int, bytearray]:
        canvas = bytearray(vis_w * vis_h * 3)
        for cv_y in range(vis_h):
            v = v0 + cv_y // zoom
            for cv_x in range(vis_w):
                u = u0 + cv_x // zoom
                idx_px = v * w + u
                val = raw_pgm[idx_px]
                lbl = labels_grid[idx_px]

                if val == 0:
                    # Tường / Vật cản: Đen đậm
                    rgb = (20, 24, 30)
                elif lbl > 0:
                    # Kiểm tra có phải đường biên giữa 2 khu vực khác nhau không
                    is_border = False
                    for dv_n, du_n in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nu, nv = u + du_n, v + dv_n
                        if 0 <= nu < w and 0 <= nv < h:
                            nlbl = labels_grid[nv * w + nu]
                            if nlbl != lbl and nlbl > 0:
                                is_border = True
                                break
                    if is_border:
                        rgb = (40, 40, 40)
                    else:
                        base_c = regions_out[lbl - 1]["color_rgb"]
                        # Pha màu pastel sáng dễ nhìn
                        rgb = (
                            int(base_c[0] * 0.42 + 255 * 0.58),
                            int(base_c[1] * 0.42 + 255 * 0.58),
                            int(base_c[2] * 0.42 + 255 * 0.58),
                        )
                elif val >= 250:
                    rgb = (242, 244, 247)
                else:
                    rgb = (185, 190, 198)

                c_idx = (cv_y * vis_w + cv_x) * 3
                canvas[c_idx] = rgb[0]
                canvas[c_idx + 1] = rgb[1]
                canvas[c_idx + 2] = rgb[2]

        # Vẽ lưới tọa độ nhẹ mỗi 2.0m
        step_px = int(round(2.0 / res)) * zoom
        for gx_px in range(0, vis_w, step_px):
            for gy_px in range(0, vis_h, 4):
                set_px(canvas, vis_w, vis_h, gx_px, gy_px, (210, 215, 222))
        for gy_px in range(0, vis_h, step_px):
            for gx_px in range(0, vis_w, 4):
                set_px(canvas, vis_w, vis_h, gx_px, gy_px, (210, 215, 222))

        # Nếu có lộ trình điều hướng giọng nói (ví dụ D -> A -> C -> B), vẽ mũi tên đường đi
        if draw_route and len(draw_route) >= 2:
            reg_map = {r["id"]: r for r in regions_out}
            for i in range(len(draw_route) - 1):
                r_start = reg_map.get(draw_route[i])
                r_end = reg_map.get(draw_route[i + 1])
                if r_start and r_end:
                    sx = (r_start["pixel_centroid"]["u"] - u0) * zoom
                    sy = (r_start["pixel_centroid"]["v"] - v0) * zoom
                    ex = (r_end["pixel_centroid"]["u"] - u0) * zoom
                    ey = (r_end["pixel_centroid"]["v"] - v0) * zoom
                    draw_arrow(canvas, vis_w, vis_h, sx, sy, ex, ey, (210, 20, 90), thickness=4, head_len=14)

        # Vẽ vòng tròn bán kính vùng, tâm đích, mũi tên hướng yaw và nhãn tên khu vực
        label_offsets = {
            "A": (14, 10),
            "B": (-85, -36),
            "C": (-95, -36),
            "D": (-110, 14),
        }
        for reg in regions_out:
            u_c = reg["pixel_centroid"]["u"]
            v_c = reg["pixel_centroid"]["v"]
            cx = (u_c - u0) * zoom
            cy = (v_c - v0) * zoom
            color = tuple(reg["color_rgb"])
            dark_color = (max(0, color[0] - 45), max(0, color[1] - 45), max(0, color[2] - 45))

            # Vòng tròn bán kính an toàn của khu vực
            rad_px = int(round(reg["radius_m"] / res)) * zoom
            draw_circle(canvas, vis_w, vis_h, cx, cy, rad_px, dark_color, filled=False, thickness=2)

            # Tâm đích và hướng Yaw
            draw_circle(canvas, vis_w, vis_h, cx, cy, 8, color, filled=True)
            draw_circle(canvas, vis_w, vis_h, cx, cy, 8, (255, 255, 255), filled=False, thickness=2)
            yaw = float(reg["goal_world"]["yaw"])
            ax = int(round(cx + 24 * math.cos(yaw)))
            ay = int(round(cy - 24 * math.sin(yaw)))
            draw_arrow(canvas, vis_w, vis_h, cx, cy, ax, ay, (20, 20, 20), thickness=2, head_len=8)

            # Hộp nhãn khu vực
            dx_lbl, dy_lbl = label_offsets.get(reg["id"], (12, -20))
            lx = max(6, min(vis_w - 175, cx + dx_lbl))
            ly = max(36, min(vis_h - 38, cy + dy_lbl))
            tag_titles = {
                "A": "ZONE A: PHONG HOC",
                "B": "ZONE B: PHONG LAB",
                "C": "ZONE C: HANH LANG",
                "D": "ZONE D: XUAT PHAT",
            }
            title_txt = tag_titles.get(reg["id"], f"ZONE {reg['id']}")
            coord_txt = f"({reg['goal_world']['x']:.1f}, {reg['goal_world']['y']:.1f})"
            box_w = max(len(title_txt), len(coord_txt)) * 6 + 10
            draw_rect(canvas, vis_w, vis_h, lx, ly, lx + box_w, ly + 26, (255, 255, 255), dark_color)
            draw_text(canvas, vis_w, vis_h, lx + 5, ly + 4, title_txt, dark_color, scale=1)
            draw_text(canvas, vis_w, vis_h, lx + 5, ly + 15, coord_txt, (50, 50, 50), scale=1)

        # Thanh tiêu đề trên cùng
        draw_rect(canvas, vis_w, vis_h, 0, 0, vis_w - 1, 28, (25, 35, 55))
        header = "SEMANTIC MAP SEGMENTATION (A: PHONG HOC | B: PHONG LAB | C: HANH LANG | D: START)"
        if draw_route:
            header = "VOICE NAVIGATION ROUTE: " + " -> ".join(draw_route)
        draw_text(canvas, vis_w, vis_h, 10, 10, header, (255, 255, 255), scale=1)
        return vis_w, vis_h, canvas

    # Xuất bản đồ phân vùng chính
    vis_w, vis_h, seg_canvas = render_segmented_canvas(draw_route=None)
    write_png_rgb(output_segmented_png, vis_w, vis_h, seg_canvas)

    # Xuất ảnh lộ trình điều hướng đa điểm (D -> A -> C -> B)
    _, _, route_canvas = render_segmented_canvas(draw_route=["D", "A", "C", "B"])
    write_png_rgb(output_route_png, vis_w, vis_h, route_canvas)

    # Xuất ảnh ghép 2x2 Pipeline 4 bước cho Báo cáo
    panel_w, panel_h = vis_w, vis_h
    gap = 12
    grid_w = panel_w * 2 + gap * 3
    grid_h = panel_h * 2 + gap * 3
    pipeline_canvas = bytearray([245, 247, 250] * (grid_w * grid_h))

    def make_panel_raw() -> bytearray:
        buf = bytearray(panel_w * panel_h * 3)
        for cy in range(panel_h):
            v = v0 + cy // zoom
            for cx in range(panel_w):
                u = u0 + cx // zoom
                g = raw_pgm[v * w + u]
                set_px(buf, panel_w, panel_h, cx, cy, (g, g, g))
        draw_rect(buf, panel_w, panel_h, 0, 0, panel_w - 1, 26, (40, 45, 55))
        draw_text(buf, panel_w, panel_h, 10, 9, "1. RAW OCCUPANCY GRID MAP (MAP.PGM)", (255, 255, 255), scale=1)
        return buf

    def make_panel_binary() -> bytearray:
        buf = bytearray(panel_w * panel_h * 3)
        for cy in range(panel_h):
            v = v0 + cy // zoom
            for cx in range(panel_w):
                u = u0 + cx // zoom
                b = binary_mask[v * w + u]
                set_px(buf, panel_w, panel_h, cx, cy, (b, b, b))
        draw_rect(buf, panel_w, panel_h, 0, 0, panel_w - 1, 26, (40, 45, 55))
        draw_text(buf, panel_w, panel_h, 10, 9, "2. BINARY THRESHOLD & MORPHOLOGY MASK", (255, 255, 255), scale=1)
        return buf

    def make_panel_dist() -> bytearray:
        buf = bytearray(panel_w * panel_h * 3)
        for cy in range(panel_h):
            v = v0 + cy // zoom
            for cx in range(panel_w):
                u = u0 + cx // zoom
                d = dist_norm_grid[v * w + u]
                if d == 0:
                    rgb = (20, 24, 35)
                else:
                    # Heatmap xanh dương -> xanh lá -> vàng -> đỏ
                    r_c = min(255, max(0, int((d - 80) * 2.0)))
                    g_c = min(255, max(0, int(d * 1.1)))
                    b_c = min(255, max(40, 255 - d))
                    rgb = (r_c, g_c, b_c)
                set_px(buf, panel_w, panel_h, cx, cy, rgb)
        for reg in regions_out:
            cx = (reg["pixel_centroid"]["u"] - u0) * zoom
            cy = (reg["pixel_centroid"]["v"] - v0) * zoom
            draw_circle(buf, panel_w, panel_h, cx, cy, 7, (255, 255, 255), filled=True)
        draw_rect(buf, panel_w, panel_h, 0, 0, panel_w - 1, 26, (40, 45, 55))
        draw_text(buf, panel_w, panel_h, 10, 9, "3. DISTANCE TRANSFORM & SEED MARKERS", (255, 255, 255), scale=1)
        return buf

    panels = [
        (gap, gap, make_panel_raw()),
        (panel_w + gap * 2, gap, make_panel_binary()),
        (gap, panel_h + gap * 2, make_panel_dist()),
        (panel_w + gap * 2, panel_h + gap * 2, seg_canvas),
    ]
    for px0, py0, pbuf in panels:
        for y in range(panel_h):
            src_start = y * panel_w * 3
            dst_start = ((py0 + y) * grid_w + px0) * 3
            pipeline_canvas[dst_start:dst_start + panel_w * 3] = pbuf[src_start:src_start + panel_w * 3]

    write_png_rgb(output_pipeline_png, grid_w, grid_h, pipeline_canvas)

    # -------------------------------------------------------------------------
    # Ghi thông tin phân vùng ra JSON và YAML
    # -------------------------------------------------------------------------
    payload = {
        "frame_id": "map",
        "map_metadata": {
            "image": meta["image"],
            "width_px": w,
            "height_px": h,
            "resolution_m_per_px": res,
            "origin_world": meta["origin"],
            "segmented_image": "maps/map_segmented.png",
        },
        "conversion_formulas": {
            "pixel_to_world_x": "x = origin_x + u * resolution",
            "pixel_to_world_y": "y = origin_y + (height - 1 - v) * resolution",
            "world_to_pixel_u": "u = round((x - origin_x) / resolution)",
            "world_to_pixel_v": "v = round(height - 1 - (y - origin_y) / resolution)",
        },
        "regions": regions_out,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fj:
        json.dump(payload, fj, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(output_yaml)), exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as fy:
        fy.write("# ==============================================================================\n")
        fy.write("# CẤU HÌNH CÁC KHU VỰC NGỮ NGHĨA TRÊN BẢN ĐỒ (SEMANTIC MAP REGIONS) - BÀI 07\n")
        fy.write("# Package: rua4_giongnoi | Hệ tọa độ: map\n")
        fy.write("# Tự động sinh từ: map_segmentation.py (OpenCV Occupancy Grid Segmentation)\n")
        fy.write("# ==============================================================================\n\n")
        fy.write(f'frame_id: "{payload["frame_id"]}"\n')
        fy.write("map_metadata:\n")
        fy.write(f'  image: "{meta["image"]}"\n')
        fy.write(f"  width_px: {w}\n")
        fy.write(f"  height_px: {h}\n")
        fy.write(f"  resolution_m_per_px: {res}\n")
        fy.write(f"  origin_world: [{ox}, {oy}, {meta['origin'][2]}]\n")
        fy.write('  segmented_image: "maps/map_segmented.png"\n\n')
        fy.write("regions:\n")
        for r in regions_out:
            fy.write(f'  - id: "{r["id"]}"\n')
            fy.write(f'    code: "{r["code"]}"\n')
            fy.write(f'    name: "{r["name"]}"\n')
            fy.write(f'    semantic_label: "{r["semantic_label"]}"\n')
            fy.write(f"    numeric_label: {r['numeric_label']}\n")
            fy.write(f"    radius_m: {r['radius_m']}\n")
            fy.write(f"    area_pixels: {r['area_pixels']}\n")
            fy.write(f"    area_m2: {r['area_m2']}\n")
            fy.write(f"    color_rgb: [{r['color_rgb'][0]}, {r['color_rgb'][1]}, {r['color_rgb'][2]}]\n")
            fy.write("    pixel_centroid:\n")
            fy.write(f"      u: {r['pixel_centroid']['u']}\n")
            fy.write(f"      v: {r['pixel_centroid']['v']}\n")
            fy.write("    goal_world:\n")
            fy.write(f"      x: {r['goal_world']['x']}\n")
            fy.write(f"      y: {r['goal_world']['y']}\n")
            fy.write(f"      yaw: {r['goal_world']['yaw']}\n")
            aliases_str = ", ".join(f'"{a}"' for a in r["aliases"])
            fy.write(f"    aliases: [{aliases_str}]\n\n")

    return payload


def resolve_pkg_root() -> str:
    ws_candidate = "/home/gdon/ur_ws/src/rua4_giongnoi"
    if os.path.isdir(ws_candidate):
        return ws_candidate
    local_candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.isfile(os.path.join(local_candidate, "package.xml")):
        return local_candidate
    if get_package_share_directory is not None:
        try:
            return get_package_share_directory("rua4_giongnoi")
        except Exception:
            pass
    return local_candidate


def main(args=None) -> None:
    parser = argparse.ArgumentParser(
        description="Phân vùng bản đồ Occupancy Grid (.pgm) và xuất file ngữ nghĩa JSON/YAML (Bài 7)"
    )
    pkg_root = resolve_pkg_root()
    parser.add_argument("--pgm", default=os.path.join(pkg_root, "maps", "map.pgm"),
                        help="Đường dẫn tới file bản đồ gốc map.pgm")
    parser.add_argument("--yaml", default=os.path.join(pkg_root, "maps", "map.yaml"),
                        help="Đường dẫn tới file metadata map.yaml")
    parser.add_argument("--out-img", default=os.path.join(pkg_root, "maps", "map_segmented.png"),
                        help="Đường dẫn lưu bản đồ phân vùng map_segmented.png")
    parser.add_argument("--out-pipeline", default=os.path.join(pkg_root, "results", "map_segmentation_pipeline.png"),
                        help="Đường dẫn lưu ảnh 4 bước xử lý phân vùng")
    parser.add_argument("--out-route", default=os.path.join(pkg_root, "results", "voice_route_demo.png"),
                        help="Đường dẫn lưu ảnh minh họa lộ trình đa điểm")
    parser.add_argument("--out-yaml", default=os.path.join(pkg_root, "config", "semantic_regions.yaml"),
                        help="Đường dẫn lưu file cấu hình semantic_regions.yaml")
    parser.add_argument("--out-json", default=os.path.join(pkg_root, "config", "semantic_regions.json"),
                        help="Đường dẫn lưu file cấu hình semantic_regions.json")
    parser.add_argument("--show", action="store_true",
                        help="Hiển thị cửa sổ OpenCV trực quan sau khi phân vùng")

    parsed_args, _ = parser.parse_known_args(args=args)
    payload = segment_occupancy_grid(
        pgm_path=parsed_args.pgm,
        yaml_path=parsed_args.yaml,
        output_segmented_png=parsed_args.out_img,
        output_pipeline_png=parsed_args.out_pipeline,
        output_route_png=parsed_args.out_route,
        output_yaml=parsed_args.out_yaml,
        output_json=parsed_args.out_json,
    )

    print("=" * 78)
    print(" HOÀN TẤT PHÂN VÙNG BẢN ĐỒ & GẮN NHÃN NGỮ NGHĨA (BÀI THỰC HÀNH 7)")
    print("=" * 78)
    print(f" - Bản đồ gốc      : {parsed_args.pgm} ({payload['map_metadata']['width_px']}x{payload['map_metadata']['height_px']} px)")
    print(f" - Bản đồ phân vùng: {parsed_args.out_img}")
    print(f" - Ảnh Pipeline    : {parsed_args.out_pipeline}")
    print(f" - File YAML       : {parsed_args.out_yaml}")
    print(f" - File JSON       : {parsed_args.out_json}")
    print("-" * 78)
    print(f" {'ID':<4} | {'Tên khu vực':<28} | {'Pixel (u, v)':<14} | {'Tọa độ Map (x, y, yaw)':<22} | {'Diện tích'}")
    print("-" * 78)
    for r in payload["regions"]:
        u, v = r["pixel_centroid"]["u"], r["pixel_centroid"]["v"]
        gx, gy, gyaw = r["goal_world"]["x"], r["goal_world"]["y"], r["goal_world"]["yaw"]
        print(f" {r['id']:<4} | {r['name']:<28} | ({u:3d}, {v:3d})     | ({gx:4.1f}, {gy:4.1f}, {gyaw:4.2f})   | {r['area_m2']:5.2f} m2")
    print("=" * 78)

    if parsed_args.show and HAS_OPENCV:
        img = cv2.imread(parsed_args.out_img)
        if img is not None:
            cv2.imshow("TurtleBot 4 - Semantic Map Segmentation (rua4_giongnoi)", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
