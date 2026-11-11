#!/usr/bin/env python3
"""Confere os MP3 gerados sem precisar ouvir: BPM medido, duração, volume e
metadados. Uso: python tools/check_audio.py"""
import glob
import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 8000  # suficiente para achar as batidas


def decode(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def measured_bpm(x):
    """BPM pelo intervalo médio entre bumbos (precisão < 0,5 BPM).

    Usa filtro casado: correlaciona o áudio com o próprio bumbo do gerador,
    assim o baixo (mesma faixa de frequência) não é confundido com batida."""
    from numpy.fft import irfft, rfft
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gen_audio
    k = gen_audio.kick()[: int(0.12 * gen_audio.SR)]
    k = k[:: gen_audio.SR // SR]  # mesma taxa de amostragem da análise
    n = len(x) + len(k)
    corr = irfft(rfft(x, n) * np.conj(rfft(k, n)), n)[: len(x)]
    corr = np.maximum(corr, 0)
    thr = 0.5 * np.max(corr)
    min_gap = int(0.2 * SR)  # 220 BPM = 0,27 s entre batidas
    peaks, i = [], 0
    while i < len(corr):
        if corr[i] > thr:
            j = i + int(np.argmax(corr[i:i + min_gap]))
            peaks.append(j)
            i = j + min_gap
        else:
            i += 1
    if len(peaks) < 8:
        return 0.0
    gaps = np.diff(peaks).astype(np.float64)
    med = np.median(gaps)
    good = gaps[(gaps > med * 0.9) & (gaps < med * 1.1)]
    return 60.0 * SR / np.mean(good)


def tags(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:format_tags=title,artist",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)["format"]
    return float(j["duration"]), j.get("tags", {})


ok = True
for kind in ("beats", "music"):
    cat_path = os.path.join(ROOT, kind, "catalog.json")
    if not os.path.exists(cat_path):
        continue
    cat = json.load(open(cat_path, encoding="utf-8"))
    print(f"== {kind}: {len(cat['tracks'])} faixas")
    for t in cat["tracks"]:
        p = os.path.join(ROOT, kind, t["id"] + ".mp3")
        x = decode(p)
        dur, tg = tags(p)
        bpm = measured_bpm(x[: SR * 60])
        peak = float(np.max(np.abs(x)))
        rms_db = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
        clip = float(np.mean(np.abs(x) > 0.99)) * 100
        # BPM medido pode sair em dobro/metade (subdivisões); aceita esses casos
        good = any(abs(bpm * k - t["bpm"]) <= 1.0 for k in (0.5, 1, 2))
        ok &= good and clip < 0.1
        print(f"  {t['id']:6s} alvo {t['bpm']:3d}  medido {bpm:6.1f} {'OK' if good else 'ERRO'}  "
              f"{dur:5.0f}s  pico {peak:.2f}  RMS {rms_db:5.1f} dB  clip {clip:.3f}%  '{tg.get('title', '?')}'")
print("RESULTADO:", "tudo OK" if ok else "há problemas")
