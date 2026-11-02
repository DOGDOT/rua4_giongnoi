#!/usr/bin/env python3
"""
Node Nhận dạng Giọng nói Đa tầng (Multi-Backend Speech Recognition Node) - Bài thực hành 7
Package: rua4_giongnoi (ROS 2 Humble)

Chức năng:
  - Tự động cân chỉnh Gain phần cứng của Card 2 (Realtek ALC257 Analog: tắt Mic Boost +20dB gây
    vỡ tiếng/nhiễu trắng 100%, đưa Capture về mức sạch ~45/63) trước khi thu âm.
  - Lọc thông cao (High-Pass Filter 80Hz khử tiếng quạt laptop) & Chuẩn hóa biên độ giọng nói (Auto-Normalize)
    về ngưỡng tối ưu ~80% giúp Google Speech-to-Text / Whisper / Vosk nhận diện tiếng Việt chính xác.
  - Hỗ trợ chuyển đổi giọng nói thành văn bản (Speech-to-Text) bằng:
      1. OpenAI Whisper (`faster-whisper` hoặc `openai-whisper`).
      2. Vosk Offline Speech Recognition (`vosk`).
      3. Google Speech Recognition (`speech_recognition` `recognize_google(language="vi-VN")`).
  - Publish câu lệnh văn bản nhận diện được lên topic `/voice_command` (`std_msgs/msg/String`).
"""

import os
import re
import sys
import time
import wave
import json
import struct
import shutil
import tempfile
import subprocess
import threading
import argparse
from typing import Optional, List, Tuple, Dict, Any

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.utilities import remove_ros_args
    from std_msgs.msg import String
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object
    String = Any


SAMPLE_VOICE_COMMANDS: List[str] = [
    "Đi đến khu vực A.",
    "Đi đến phòng thí nghiệm.",
    "Di chuyển từ khu vực A đến khu vực B.",
    "Đi từ phòng A qua hành lang C rồi đến phòng B.",
    "Đi đến khu vực D.",
    "Đi đến khu vực Z.",
]


class SpeechEngine:
    """Bộ thu âm Microphone đa tầng (Card 2 ALC257 / PulseAudio) kèm lọc nhiễu & chuẩn hóa âm lượng."""

    def __init__(self, preferred_engine: str = "auto", whisper_model_size: str = "base",
                 vosk_model_path: Optional[str] = None, language: str = "vi") -> None:
        self.preferred_engine = preferred_engine.lower()
        self.whisper_model_size = whisper_model_size
        self.vosk_model_path = vosk_model_path
        self.language = language
        self.active_engine = "google_sr"
        self._whisper_model = None
        self._whisper_type = None
        self._vosk_model = None
        self.last_record_backend: str = "none"
        self.last_peak_percent: float = 0.0

        self._init_engine()

    def _init_engine(self) -> None:
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

        if self.preferred_engine in ("auto", "google"):
            try:
                import speech_recognition as sr  # type: ignore
                _ = sr.Recognizer()
                self.active_engine = "google_sr"
                return
            except Exception:
                pass

        self.active_engine = "google_sr"

    @staticmethod
    def _get_audio_env() -> Dict[str, str]:
        """Đảm bảo biến môi trường PulseAudio/PipeWire/DBus luôn kết nối đúng tới Audio Server của user."""
        env = os.environ.copy()
        uid = os.getuid()
        runtime_dir = f"/run/user/{uid}"
        if os.path.isdir(runtime_dir):
            env.setdefault("XDG_RUNTIME_DIR", runtime_dir)
            pulse_sock = f"{runtime_dir}/pulse/native"
            if os.path.exists(pulse_sock):
                env.setdefault("PULSE_SERVER", f"unix:{pulse_sock}")
            bus_sock = f"{runtime_dir}/bus"
            if os.path.exists(bus_sock):
                env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={bus_sock}")
        return env

    @staticmethod
    def _wrap_cmd(cmd_args: List[str]) -> List[List[str]]:
        """Trả về danh sách lệnh thử trực tiếp và qua distrobox-host-exec (nếu chạy trong container)."""
        candidates = []
        if shutil.which(cmd_args[0]):
            candidates.append(cmd_args)
        if shutil.which("distrobox-host-exec"):
            candidates.append(["distrobox-host-exec"] + cmd_args)
        return candidates

    def _calibrate_card2_mixer(self) -> None:
        """
        Tự động cân chỉnh ALSA Mixer trên Card 2 (Realtek ALC257):
        - Tắt Internal Mic Boost (0dB) để tránh hiện tượng khuếch đại quá mức (+50dB) gây bão hòa nhiễu trắng 100%.
        - Đặt Capture ở mức 48/63 (+18.75dB) để giọng nói thu vào trong trẻo, không bị méo (clipping = 0%).
        """
        env = self._get_audio_env()
        mixer_cmds = [
            ["pactl", "set-source-mute", "@DEFAULT_SOURCE@", "0"],
            ["pactl", "set-source-volume", "@DEFAULT_SOURCE@", "24%"],
            ["amixer", "-c", "2", "sset", "Internal Mic Boost", "0"],
            ["amixer", "-c", "2", "sset", "Mic Boost", "0"],
            ["amixer", "-c", "2", "sset", "Capture", "48", "cap"],
        ]
        for mcmd in mixer_cmds:
            for full_cmd in self._wrap_cmd(mcmd):
                try:
                    subprocess.run(
                        full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=1.2, env=env
                    )
                    break
                except Exception:
                    continue

    def _detect_alsa_capture_devices(self) -> List[str]:
        """Quét `arecord -l` để ưu tiên Card 2 (`plughw:2,0` - ALC257 Analog) và các Card thu âm thực tế."""
        devices: List[str] = ["plughw:2,0"]
        env = self._get_audio_env()
        for base_cmd in self._wrap_cmd(["arecord", "-l"]):
            try:
                res = subprocess.run(
                    base_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=2.5, env=env
                )
                for line in res.stdout.splitlines():
                    m = re.search(r"card\s+(\d+):.*device\s+(\d+):", line, re.IGNORECASE)
                    if m:
                        dev_str = f"plughw:{m.group(1)},{m.group(2)}"
                        if dev_str not in devices:
                            devices.append(dev_str)
            except Exception:
                pass
        return devices

    @staticmethod
    def _process_and_normalize_pcm(raw_pcm: bytes, sample_rate: int = 16000) -> Tuple[bytes, float]:
        """
        Lọc nhiễu tần số thấp (High-pass ~85Hz khử tiếng ù quạt máy tính + khử DC offset)
        và tự động chuẩn hóa biên độ giọng nói (Auto-Gain Normalization) giúp nhận diện rõ câu lệnh.
        Trả về (processed_pcm_bytes, raw_peak_percent).
        """
        if len(raw_pcm) < 4:
            return raw_pcm, 0.0
        count = len(raw_pcm) // 2
        samples = list(struct.unpack(f"<{count}h", raw_pcm[:count * 2]))

        # 1. Bộ lọc thông cao bậc 1 (1st-order High-Pass Filter fc ~ 85Hz tại fs=16000Hz -> alpha = 0.967)
        alpha = 0.967
        filtered: List[float] = [0.0] * count
        prev_x = float(samples[0])
        prev_y = 0.0
        for i, x_val in enumerate(samples):
            xf = float(x_val)
            yf = alpha * (prev_y + xf - prev_x)
            filtered[i] = yf
            prev_x = xf
            prev_y = yf

        # Bỏ 0.08 giây đầu tiên (tránh tiếng "bụp" khi vừa mở cổng âm thanh)
        skip_n = min(count // 10, int(0.08 * sample_rate))
        for i in range(skip_n):
            filtered[i] = 0.0

        raw_peak = max((abs(v) for v in filtered), default=0.0)
        raw_peak_pct = round(min(100.0, (raw_peak / 32768.0) * 100.0), 1)

        # 2. Tự động chuẩn hóa biên độ (Auto-Normalize) về ngưỡng 24000 (~73% Full-Scale)
        if raw_peak > 150.0:
            gain = min(10.0, 24000.0 / raw_peak)
        else:
            gain = 1.0

        out_samples = [
            max(-32767, min(32767, int(round(v * gain))))
            for v in filtered
        ]
        processed_bytes = struct.pack(f"<{count}h", *out_samples)
        return processed_bytes, raw_peak_pct

    @staticmethod
    def _save_pcm_to_wav(wav_path: str, raw_pcm: bytes, sample_rate: int = 16000) -> bool:
        if len(raw_pcm) < 3200:
            return False
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_pcm)
        return os.path.isfile(wav_path) and os.path.getsize(wav_path) > 3200

    def record_microphone_wav(self, duration_sec: float = 4.0, sample_rate: int = 16000) -> Optional[str]:
        """
        Thu âm giọng nói từ Microphone ra file WAV 16kHz 16-bit Mono đã lọc nhiễu & chuẩn hóa âm lượng.
        """
        tmp_wav = os.path.join(tempfile.gettempdir(), "rua4_voice_cmd.wav")
        try:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        except Exception:
            pass

        # Cân chỉnh Gain Card 2 trước khi thu để không bị bão hòa nhiễu trắng (Clipping)
        self._calibrate_card2_mixer()

        num_bytes = int(sample_rate * duration_sec * 2)
        dur_int = max(1, int(round(duration_sec)))
        env = self._get_audio_env()

        raw_stream_cmds: List[Tuple[str, List[str]]] = []

        # Tầng 1: ALSA trực tiếp vào Card 2 (`plughw:2,0` - ALC257 Analog) với Gain sạch đã cân chỉnh
        for hw_dev in self._detect_alsa_capture_devices():
            for cmd in self._wrap_cmd([
                "arecord", "-q", "-D", hw_dev, "-t", "raw", "-f", "S16_LE",
                "-r", str(sample_rate), "-c", "1", "-d", str(dur_int)
            ]):
                raw_stream_cmds.append((f"Card 2 ALSA ({hw_dev})", cmd))

        # Tầng 2: PulseAudio / PipeWire `parecord --raw`
        for cmd in self._wrap_cmd([
            "parecord", "--raw", "--channels=1", f"--rate={sample_rate}", "--format=s16le"
        ]):
            raw_stream_cmds.append(("parecord (Card 2 PulseAudio)", cmd))

        # Tầng 3: ALSA qua plugin `pulse` hoặc `default`
        for dev_name in ("pulse", "pipewire", "default"):
            for cmd in self._wrap_cmd([
                "arecord", "-q", "-D", dev_name, "-t", "raw", "-f", "S16_LE",
                "-r", str(sample_rate), "-c", "1", "-d", str(dur_int)
            ]):
                raw_stream_cmds.append((f"arecord -D {dev_name}", cmd))

        fallback_zero_pcm: Optional[Tuple[str, bytes]] = None

        for backend_label, cmd in raw_stream_cmds:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
                raw_pcm = proc.stdout.read(num_bytes) if proc.stdout else b""
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    proc.kill()

                if len(raw_pcm) >= 3200:
                    clean_pcm, peak_pct = self._process_and_normalize_pcm(raw_pcm, sample_rate)
                    if peak_pct > 0.0:
                        if self._save_pcm_to_wav(tmp_wav, clean_pcm, sample_rate):
                            self.last_record_backend = backend_label
                            self.last_peak_percent = peak_pct
                            return tmp_wav
                    elif fallback_zero_pcm is None:
                        fallback_zero_pcm = (backend_label, clean_pcm)
            except Exception:
                continue

        if fallback_zero_pcm is not None:
            backend_label, clean_pcm = fallback_zero_pcm
            if self._save_pcm_to_wav(tmp_wav, clean_pcm, sample_rate):
                self.last_record_backend = backend_label
                self.last_peak_percent = 0.0
                return tmp_wav

        return None

    def transcribe_file(self, wav_path: str) -> str:
        """Chuyển đổi file âm thanh .wav thành văn bản tiếng Việt."""
        if not wav_path or not os.path.isfile(wav_path):
            return ""

        # 1. Whisper (faster-whisper / openai-whisper)
        if self._whisper_model is not None:
            try:
                if self._whisper_type == "faster_whisper":
                    segments, _ = self._whisper_model.transcribe(
                        wav_path, language=self.language, beam_size=5
                    )
                    text = " ".join(seg.text.strip() for seg in segments).strip()
                    if text:
                        return text
                else:
                    result = self._whisper_model.transcribe(wav_path, language=self.language, fp16=False)
                    text = str(result.get("text", "")).strip()
                    if text:
                        return text
            except Exception:
                pass

        # 2. Vosk Offline
        if self._vosk_model is not None:
            try:
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
                    text = " ".join(text_parts).strip()
                    if text:
                        return text
            except Exception:
                pass

        # 3. Google SpeechRecognition (`speech_recognition` -> `recognize_google(language="vi-VN")`)
        try:
            import speech_recognition as sr  # type: ignore
            recognizer = sr.Recognizer()
            recognizer.dynamic_energy_threshold = False
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
            res = recognizer.recognize_google(audio_data, language="vi-VN", show_all=True)
            if isinstance(res, dict) and "alternative" in res and res["alternative"]:
                # Ưu tiên câu có chứa từ khóa khu vực/phòng/hành lang nếu có trong danh sách phương án
                alts = [str(item.get("transcript", "")).strip() for item in res["alternative"] if item.get("transcript")]
                for alt in alts:
                    low = alt.lower()
                    if any(k in low for k in ("khu", "phòng", "hành lang", "đến", "từ", "qua", "về", "a", "b", "c", "d")):
                        return alt
                if alts:
                    return alts[0]
            elif isinstance(res, str) and res.strip():
                return res.strip()
        except Exception:
            pass

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

        self.get_logger().info("Đang khởi tạo bộ thu âm & nhận dạng giọng nói (Card 2 ALC257 + Auto-Gain + Google STT)...")
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
        print(" - Các khu vực hợp lệ  : A (Phòng học) | B (Phòng thí nghiệm) | C (Hành lang) | D (Khu kho)")
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
                    print("⚠️  Không tìm thấy thiết bị thu âm Microphone.")
                    continue
                print(
                    f"🎙️  Đã thu âm thành công qua [{self.engine.last_record_backend}] "
                    f"(Mức âm lượng gốc: {self.engine.last_peak_percent:.1f}%). Đang nhận dạng..."
                )
                recognized = self.engine.transcribe_file(wav_path)
                if recognized:
                    print(f'✅ Kết quả nhận dạng giọng nói: "{recognized}"')
                    self.publish_voice_text(recognized)
                else:
                    print("⚠️  Chưa nhận dạng rõ câu nói. Hãy nói to rõ hơn hoặc chọn phím [1..6].")
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
                        help="Phát một câu lệnh duy nhất rồi kết thúc")
    parser.add_argument("--audio-file", default=None,
                        help="Nhận dạng trực tiếp từ file âm thanh .wav")
    parser.add_argument("--test-mic", action="store_true",
                        help="Kiểm tra nhanh thu âm Microphone 2 giây rồi in thông số")

    clean_argv = remove_ros_args(args=sys.argv)[1:] if HAS_ROS2 else sys.argv[1:]
    parsed, _ = parser.parse_known_args(clean_argv)

    if parsed.test_mic:
        eng = SpeechEngine(preferred_engine=parsed.engine)
        wav = eng.record_microphone_wav(duration_sec=2.0)
        if wav:
            print(f"[PASS] Thu âm thành công: {wav} ({os.path.getsize(wav)} bytes)")
            print(f"       Backend: {eng.last_record_backend} | Raw Peak Volume: {eng.last_peak_percent:.1f}%")
            sys.exit(0)
        else:
            print("[FAIL] Không thu âm được từ Microphone.")
            sys.exit(1)

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
