#!/usr/bin/env python3
"""
Node Nhận dạng Giọng nói (Speech Recognition Node) - Bài thực hành 7
Package: rua4_giongnoi (ROS 2 Humble)

Chức năng:
  - Thu nhận giọng nói tiếng Việt / tiếng Anh từ Microphone (qua ALSA `arecord` hoặc `sounddevice`)
    hoặc từ file âm thanh `.wav` (`--audio-file`).
  - Hỗ trợ chuyển đổi giọng nói thành văn bản (Speech-to-Text) bằng:
      1. OpenAI Whisper (`faster-whisper` hoặc `openai-whisper`) - Độ chính xác tiếng Việt rất cao.
      2. Vosk Offline Speech Recognition (`vosk`) - Nhẹ, chạy hoàn toàn offline.
      3. Google Web Speech API (`speech_recognition`) - Dự phòng khi có mạng.
      4. Giao diện Phím tắt & Giọng nói tương tác (Interactive Voice/CLI Console) để thu âm bằng
         phím Enter hoặc kích hoạt nhanh các câu lệnh mẫu trong quá trình kiểm thử & quay video demo.
  - Publish câu lệnh văn bản nhận diện được lên topic `/voice_command` (`std_msgs/msg/String`).
"""

import os
import sys
import time
import wave
import json
import tempfile
import subprocess
import threading
import argparse
from typing import Optional, List

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.utilities import remove_ros_args
    from std_msgs.msg import String
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object


# Danh sách câu lệnh mẫu phục vụ kiểm thử nhanh và demo đầy đủ 5 tiêu chí của Bài thực hành 7
SAMPLE_VOICE_COMMANDS: List[str] = [
    "Đi đến khu vực A.",
    "Đi đến phòng thí nghiệm.",
    "Di chuyển từ khu vực A đến khu vực B.",
    "Đi từ phòng A qua hành lang C rồi đến phòng B.",
    "Đi đến khu vực Z.",
    "Đi đến trạm xuất phát.",
]


class SpeechEngine:
    """Bộ nhận dạng giọng nói đa nền tảng hỗ trợ Whisper, Vosk và SpeechRecognition."""

    def __init__(self, preferred_engine: str = "auto", whisper_model_size: str = "base",
                 vosk_model_path: Optional[str] = None, language: str = "vi") -> None:
        self.preferred_engine = preferred_engine.lower()
        self.whisper_model_size = whisper_model_size
        self.vosk_model_path = vosk_model_path
        self.language = language
        self.active_engine = "cli"
        self._whisper_model = None
        self._whisper_type = None
        self._vosk_model = None

        self._init_engine()

    def _init_engine(self) -> None:
        # 1. Thử khởi tạo faster-whisper hoặc openai-whisper
        if self.preferred_engine in ("auto", "whisper"):
            try:
                from faster_whisper import WhisperModel  # type: ignore
                self._whisper_model = WhisperModel(
                    self.whisper_model_size, device="cpu", compute_type="int8"
                )
                self._whisper_type = "faster_whisper"
                self.active_engine = "whisper"
                return
            except Exception:
                pass

            try:
                import whisper  # type: ignore
                self._whisper_model = whisper.load_model(self.whisper_model_size)
                self._whisper_type = "openai_whisper"
                self.active_engine = "whisper"
                return
            except Exception:
                pass

        # 2. Thử khởi tạo Vosk
        if self.preferred_engine in ("auto", "vosk"):
            try:
                from vosk import Model, SetLogLevel  # type: ignore
                SetLogLevel(-1)
                if self.vosk_model_path and os.path.isdir(self.vosk_model_path):
                    self._vosk_model = Model(self.vosk_model_path)
                else:
                    self._vosk_model = Model(lang=self.language)
                self.active_engine = "vosk"
                return
            except Exception:
                pass

        # 3. Thử khởi tạo SpeechRecognition (Google STT)
        if self.preferred_engine in ("auto", "google"):
            try:
                import speech_recognition as sr  # type: ignore
                _ = sr.Recognizer()
                self.active_engine = "google"
                return
            except Exception:
                pass

        self.active_engine = "cli"

    def record_microphone_wav(self, duration_sec: float = 4.0, sample_rate: int = 16000) -> Optional[str]:
        """Thu âm giọng nói từ Microphone ra file WAV 16kHz Mono bằng arecord hoặc sounddevice."""
        tmp_wav = os.path.join(tempfile.gettempdir(), "rua4_voice_cmd.wav")

        # Ưu tiên công cụ ALSA `arecord` có sẵn trên mọi bản phân phối Linux
        try:
            cmd = [
                "arecord",
                "-q",
                "-f", "S16_LE",
                "-r", str(sample_rate),
                "-c", "1",
                "-d", str(max(1, int(round(duration_sec)))),
                tmp_wav,
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=duration_sec + 3.0)
            if res.returncode == 0 and os.path.isfile(tmp_wav) and os.path.getsize(tmp_wav) > 1000:
                return tmp_wav
        except Exception:
            pass

        # Dự phòng bằng thư viện sounddevice nếu có
        try:
            import sounddevice as sd  # type: ignore
            import numpy as np
            frames = int(duration_sec * sample_rate)
            recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
            sd.wait()
            with wave.open(tmp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(recording.tobytes())
            return tmp_wav
        except Exception:
            return None

    def transcribe_file(self, wav_path: str) -> str:
        """Chuyển đổi file âm thanh .wav thành văn bản tiếng Việt."""
        if not wav_path or not os.path.isfile(wav_path):
            return ""

        if self.active_engine == "whisper" and self._whisper_model is not None:
            if self._whisper_type == "faster_whisper":
                segments, _ = self._whisper_model.transcribe(
                    wav_path, language=self.language, beam_size=5
                )
                return " ".join(seg.text.strip() for seg in segments).strip()
            else:
                result = self._whisper_model.transcribe(wav_path, language=self.language, fp16=False)
                return str(result.get("text", "")).strip()

        if self.active_engine == "vosk" and self._vosk_model is not None:
            from vosk import KaldiRecognizer  # type: ignore
            with wave.open(wav_path, "rb") as wf:
                rec = KaldiRecognizer(self._vosk_model, wf.getframerate())
                text_parts = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        part = json.loads(rec.Result()).get("text", "")
                        if part:
                            text_parts.append(part)
                final_part = json.loads(rec.FinalResult()).get("text", "")
                if final_part:
                    text_parts.append(final_part)
                return " ".join(text_parts).strip()

        if self.active_engine == "google":
            try:
                import speech_recognition as sr  # type: ignore
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                return recognizer.recognize_google(audio_data, language="vi-VN").strip()
            except Exception:
                return ""

        return ""


class SpeechRecognitionNode(Node):
    """ROS 2 Node tiếp nhận giọng nói và publish câu lệnh văn bản lên `/voice_command`."""

    def __init__(self, engine_name: str = "auto", whisper_model: str = "base",
                 vosk_model_path: Optional[str] = None, record_seconds: float = 4.0,
                 once_command: Optional[str] = None, audio_file: Optional[str] = None) -> None:
        super().__init__("speech_recognition_node")
        self.pub_cmd = self.create_publisher(String, "/voice_command", 10)
        self.sub_status = self.create_subscription(
            String, "/voice_nav_status", self._on_nav_status, 10
        )
        self.record_seconds = record_seconds
        self.once_command = once_command
        self.audio_file = audio_file
        self.last_status = "IDLE - Đang chờ câu lệnh giọng nói..."

        self.get_logger().info("Đang khởi tạo bộ nhận dạng giọng nói (Whisper / Vosk)...")
        self.engine = SpeechEngine(
            preferred_engine=engine_name,
            whisper_model_size=whisper_model,
            vosk_model_path=vosk_model_path,
            language="vi",
        )
        self.get_logger().info(
            f"Bộ nhận dạng sẵn sàng! Engine hoạt động: [{self.engine.active_engine.upper()}] | Topic: /voice_command"
        )

    def _on_nav_status(self, msg: String) -> None:
        self.last_status = msg.data

    def publish_voice_text(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        msg = String()
        msg.data = cleaned
        self.pub_cmd.publish(msg)
        self.get_logger().info(f'🎤 Đã phát câu lệnh lên /voice_command: "{cleaned}"')

    def run_interactive_loop(self) -> None:
        """Vòng lặp giao diện điều khiển bằng giọng nói & phím tắt trực quan trên Terminal."""
        time.sleep(0.6)

        if self.audio_file:
            self.get_logger().info(f"Đang nhận dạng từ file âm thanh: {self.audio_file}")
            text = self.engine.transcribe_file(self.audio_file)
            if text:
                self.publish_voice_text(text)
            else:
                self.get_logger().warn("Không nhận dạng được văn bản từ file âm thanh.")
            return

        if self.once_command:
            self.publish_voice_text(self.once_command)
            return

        print("\n" + "=" * 78)
        print(" 🎙️  HỆ THỐNG NHẬN DẠNG GIỌNG NÓI & ĐIỀU HƯỚNG TURTLEBOT 4 (BÀI THỰC HÀNH 7)")
        print("=" * 78)
        print(f" - Engine STT hiện tại : {self.engine.active_engine.upper()}")
        print(" - Topic xuất câu lệnh : /voice_command (std_msgs/msg/String)")
        print(" - Các khu vực hợp lệ  : A (Phòng học) | B (Phòng thí nghiệm) | C (Hành lang) | D (Trạm xuất phát)")
        print("-" * 78)
        print(" HƯỚNG DẪN SỬ DỤNG:")
        print(f"   • Nhấn [ENTER] trống : Thu âm giọng nói trực tiếp từ Microphone ({self.record_seconds:.0f} giây)")
        print("   • Nhập số [1..6]     : Phát nhanh câu lệnh giọng nói mẫu:")
        for idx, cmd in enumerate(SAMPLE_VOICE_COMMANDS, start=1):
            print(f'       [{idx}] "{cmd}"')
        print("   • Hoặc gõ trực tiếp bất kỳ câu lệnh tiếng Việt nào (ví dụ: Đi từ A qua C rồi đến B)")
        print("   • Nhập 'q' để thoát.")
        print("=" * 78)

        while rclpy.ok():
            try:
                user_in = input("\n🎤 [Nhấn ENTER để nói qua Mic | Chọn 1-6 | Nhập câu lệnh]: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if user_in.lower() in ("q", "quit", "exit"):
                break

            if user_in == "":
                print(f"🔴 Đang thu âm Microphone trong {self.record_seconds:.0f} giây... Hãy nói câu lệnh!")
                wav_path = self.engine.record_microphone_wav(duration_sec=self.record_seconds)
                if not wav_path:
                    print("⚠️  Không tìm thấy thiết bị thu âm Microphone hoặc chưa cấp quyền audio.")
                    print("👉 Bạn có thể chọn phím [1..6] hoặc nhập trực tiếp câu lệnh tiếng Việt.")
                    continue
                print("⏳ Đang nhận dạng giọng nói (Speech-to-Text)...")
                recognized = self.engine.transcribe_file(wav_path)
                if recognized:
                    print(f'✅ Kết quả nhận dạng giọng nói: "{recognized}"')
                    self.publish_voice_text(recognized)
                else:
                    print("⚠️  Không nghe rõ hoặc chưa cài đặt model STT offline. Hãy thử lại hoặc nhập câu lệnh.")
                continue

            if user_in.isdigit() and 1 <= int(user_in) <= len(SAMPLE_VOICE_COMMANDS):
                chosen = SAMPLE_VOICE_COMMANDS[int(user_in) - 1]
                print(f'🎙️  [Giọng nói mẫu #{user_in}]: "{chosen}"')
                self.publish_voice_text(chosen)
                continue

            self.publish_voice_text(user_in)


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="ROS 2 Speech Recognition Node cho Bài thực hành 7")
    parser.add_argument("--engine", default="auto", choices=["auto", "whisper", "vosk", "google", "cli"],
                        help="Chọn bộ nhận dạng giọng nói (mặc định: auto)")
    parser.add_argument("--whisper-model", default="base",
                        help="Kích thước model Whisper: tiny, base, small (mặc định: base)")
    parser.add_argument("--vosk-model-path", default=None,
                        help="Đường dẫn thư mục model Vosk tiếng Việt")
    parser.add_argument("--record-seconds", type=float, default=4.0,
                        help="Thời lượng thu âm mỗi câu lệnh qua Microphone (giây)")
    parser.add_argument("--once", default=None,
                        help="Phát một câu lệnh duy nhất rồi kết thúc (phục vụ kiểm thử tự động)")
    parser.add_argument("--audio-file", default=None,
                        help="Nhận dạng trực tiếp từ file âm thanh .wav")

    clean_argv = remove_ros_args(args=sys.argv)[1:] if HAS_ROS2 else sys.argv[1:]
    parsed, _ = parser.parse_known_args(clean_argv)

    if not HAS_ROS2:
        print("Lỗi: Cần chạy trong môi trường ROS 2 Humble (source /opt/ros/humble/setup.bash).")
        sys.exit(1)

    rclpy.init(args=args)
    node = SpeechRecognitionNode(
        engine_name=parsed.engine,
        whisper_model=parsed.whisper_model,
        vosk_model_path=parsed.vosk_model_path,
        record_seconds=parsed.record_seconds,
        once_command=parsed.once,
        audio_file=parsed.audio_file,
    )

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run_interactive_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
