"""用 Kokoro 把 index.html 句庫（RAW）的英文句子轉成 mp3。

用法（在專案根目錄執行）：
  .tts/venv/Scripts/python.exe tools/make_audio.py --sample "I’d like to check one bag."
      每個候選聲音各產生一個試聽檔到 .tts/samples/
  .tts/venv/Scripts/python.exe tools/make_audio.py --voice af_heart
      產生 .tts/audio/<id>.mp3；已存在的檔案會跳過，加 --force 全部重做
  .tts/venv/Scripts/python.exe tools/make_audio.py --embed
      把 .tts/audio 的 mp3 以 base64 寫進 index.html 最後的 #audio-data 區塊（整段覆蓋）

需要：.tts/venv（kokoro-onnx）、.tts/models（kokoro-v1.0.onnx、voices-v1.0.bin）、ffmpeg。
"""
import argparse
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

ROOT = Path(__file__).resolve().parent.parent
TTS = ROOT / ".tts"
CANDIDATES = ["af_heart", "af_bella", "af_nicole", "am_michael", "am_fenrir", "am_puck"]


def load_sentences():
    """讀出 RAW 陣列裡的每一句，回傳 [(id, 英文)]，id 即陣列索引。"""
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    body = re.search(r"const RAW = \[(.*?)\n\];", html, re.S).group(1)
    rows = [json.loads(line.strip().rstrip(",")) for line in body.splitlines() if line.strip().startswith("[")]
    return [(i, r[2]) for i, r in enumerate(rows)]


def to_mp3(samples, rate, out):
    """前後各留 0.15 秒靜音，轉成單聲道 48 kbps mp3。"""
    pad = np.zeros(int(rate * 0.15), dtype=np.float32)
    pcm = (np.clip(np.concatenate([pad, samples, pad]), -1, 1) * 32767).astype("<i2").tobytes()
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "-",
         "-codec:a", "libmp3lame", "-b:a", "48k", str(out)],
        input=pcm, check=True)


BEGIN, END = "<!-- AUDIO:BEGIN", "<!-- AUDIO:END -->"


def embed():
    """一句一行寫成 JSON，放在 </body> 前，網頁播放時才讀取。"""
    path = ROOT / "index.html"
    html = path.read_text(encoding="utf-8")
    ids = [i for i, _ in load_sentences() if (TTS / "audio" / f"{i}.mp3").exists()]
    lines = ",\n".join(
        f'"{i}":"{base64.b64encode((TTS / "audio" / f"{i}.mp3").read_bytes()).decode()}"' for i in ids)
    block = (f"{BEGIN}：由 tools/make_audio.py --embed 產生，請勿手動修改 -->\n"
             f'<script type="application/json" id="audio-data">{{\n{lines}\n}}</script>\n{END}\n')
    if BEGIN in html:
        html = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n", lambda m: block, html, flags=re.S)
    else:
        html = html.replace("</body>", block + "</body>")
    path.write_text(html, encoding="utf-8", newline="\n")
    print(f"已嵌入 {len(ids)} 句，index.html 共 {path.stat().st_size // 1024} KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=0.95)
    ap.add_argument("--sample", help="只用這句英文產生各聲音的試聽檔")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--embed", action="store_true")
    a = ap.parse_args()
    if a.embed:
        return embed()

    kokoro = Kokoro(str(TTS / "models/kokoro-v1.0.onnx"), str(TTS / "models/voices-v1.0.bin"))
    say = lambda text, voice: kokoro.create(text.replace("’", "'"), voice=voice, speed=a.speed, lang="en-us")

    if a.sample:
        for v in CANDIDATES:
            samples, rate = say(a.sample, v)
            to_mp3(samples, rate, TTS / "samples" / f"{v}.mp3")
            print("試聽檔", v)
        return

    for i, en in load_sentences():
        out = TTS / "audio" / f"{i}.mp3"
        if out.exists() and not a.force:
            continue
        samples, rate = say(en, a.voice)
        to_mp3(samples, rate, out)
        print(i, en)


if __name__ == "__main__":
    sys.exit(main())
