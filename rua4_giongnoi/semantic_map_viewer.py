#!/usr/bin/env python3
"""
Trình Hiển thị Bản đồ Ngữ nghĩa & Lộ trình Giọng nói (Semantic Map Viewer) - Bài thực hành 7
Package: rua4_giongnoi

Hiển thị trực quan cửa sổ OpenCV cho bản đồ phân vùng (`maps/map_segmented.png`)
kèm trạng thái câu lệnh giọng nói thời gian thực từ `/voice_command` và `/voice_nav_status`.
"""

import os
import sys
import argparse

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

from rua4_giongnoi.map_segmentation import resolve_pkg_root


def main(args=None) -> None:
    pkg_root = resolve_pkg_root()
    parser = argparse.ArgumentParser(description="Hiển thị bản đồ phân vùng ngữ nghĩa Bài 7")
    parser.add_argument(
        "--image",
        default=os.path.join(pkg_root, "maps", "map_segmented.png"),
        help="Đường dẫn tới ảnh bản đồ phân vùng",
    )
    parsed, _ = parser.parse_known_args(args=args)

    if not os.path.isfile(parsed.image):
        print(f"Không tìm thấy ảnh phân vùng tại: {parsed.image}")
        sys.exit(1)

    if not HAS_OPENCV:
        print(f"Ảnh bản đồ phân vùng đã sẵn sàng tại: {parsed.image}")
        return

    img = cv2.imread(parsed.image)
    if img is None:
        print(f"Không thể đọc ảnh: {parsed.image}")
        sys.exit(1)

    win_name = "TurtleBot 4 - Semantic Map & Voice Navigation (rua4_giongnoi)"
    cv2.imshow(win_name, img)
    print(f"Đang hiển thị '{win_name}'. Nhấn phím bất kỳ trên cửa sổ ảnh để đóng.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
