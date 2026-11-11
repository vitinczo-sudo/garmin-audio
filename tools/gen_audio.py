#!/usr/bin/env python3
"""Gera TODO o áudio dos apps Smart Run Beats e Smart Workout Music.

Todo o som é sintetizado aqui, do zero (numpy): nenhuma amostra, loop ou
música de terceiros é usada. Por isso o áudio é original e pode ser
distribuído livremente (ver LICENSE na raiz do repositório).

Saída (pasta do site / GitHub Pages):
  beats/<id>.mp3  + beats/catalog.json   (batidas de treino em BPM exato)
  music/<id>.mp3  + music/catalog.json   (faixas instrumentais)

Uso:  python tools/gen_audio.py [--only beats|music] [--seconds N]
Requer: numpy e ffmpeg no PATH.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

SR = 32000  # 32 kHz mono: bom para fone de corrida e arquivos menores
BASE_URL = "https://vitinczo-sudo.github.io/garmin-audio"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOTE = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def hz(name, octave):
    """Frequência da nota (A4 = 440 Hz)."""
    return 440.0 * 2 ** ((NOTE[name] + 12 * (octave + 1) - 69) / 12)


def env(n, attack, release, sustain=1.0):
    """Envelope linear de ataque/release (em amostras)."""
    e = np.full(n, sustain, dtype=np.float32)
    a = min(int(attack), n)
    r = min(int(release), n - a)
    if a > 0:
        e[:a] = np.linspace(0, sustain, a, dtype=np.float32)
    if r > 0:
        e[n - r:] = np.linspace(sustain, 0, r, dtype=np.float32)
    return e


def tone(freq, dur, harmonics, decay=None):
    """Onda com harmônicos limitados (sem aliasing, sem precisar de filtro)."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros(n, dtype=np.float32)
    for k, amp in harmonics:
        if freq * k < SR / 2.2:
            out += amp * np.sin(2 * np.pi * freq * k * t)
    if decay:
        out *= np.exp(-t / decay)
    return out


SAW = [(k, 1.0 / k) for k in range(1, 9)]
SOFT_SAW = [(k, 1.0 / (k * k ** 0.5)) for k in range(1, 7)]
TRI = [(1, 1.0), (3, 1 / 9), (5, 1 / 25), (7, 1 / 49)]
SQUARE = [(k, 1.0 / k) for k in (1, 3, 5, 7, 9)]


# ---------------------------------------------------------------- bateria
def kick():
    n = int(0.32 * SR)
    t = np.arange(n) / SR
    f = 45 + 95 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return (np.sin(ph) * np.exp(-t / 0.13)).astype(np.float32)


def snare(rng):
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n).astype(np.float32)
    noise = np.diff(noise, prepend=0) * 0.6
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    return ((noise * np.exp(-t / 0.07)) * 0.55 + body * 0.5).astype(np.float32)


def hat(rng, open_=False):
    n = int((0.18 if open_ else 0.05) * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n).astype(np.float32)
    noise = np.diff(np.diff(noise, prepend=0), prepend=0) * 0.25
    return (noise * np.exp(-t / (0.06 if open_ else 0.012))).astype(np.float32)


def add(buf, sig, pos, gain=1.0):
    pos = int(pos)
    if pos >= len(buf):
        return
    end = min(len(buf), pos + len(sig))
    buf[pos:end] += sig[: end - pos] * gain


# ---------------------------------------------------------------- arranjo
def chord_notes(root, minor):
    third = 3 if minor else 4
    names = list(NOTE)
    r = NOTE[root]
    return [names[r], names[(r + third) % 12], names[(r + 7) % 12]]


def render(bpm, seconds, progression, seed, style):
    """style: 'beats' (batida de treino) ou 'music' (faixa com melodia)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    drums = np.zeros(n, np.float32)
    bass = np.zeros(n, np.float32)
    pad = np.zeros(n, np.float32)
    lead = np.zeros(n, np.float32)
    beat = 60.0 / bpm * SR
    bar = 4 * beat
    nbars = int(n / bar) + 1
    K, S, H, HO = kick(), snare(rng), hat(rng), hat(rng, True)

    # Melodia: frases de 2 compassos geradas a partir da escala do acorde.
    scale_cache = {}

    def scale(root, minor):
        key = (root, minor)
        if key not in scale_cache:
            steps = [0, 2, 3, 5, 7, 8, 10] if minor else [0, 2, 4, 5, 7, 9, 11]
            names = list(NOTE)
            scale_cache[key] = [names[(NOTE[root] + s) % 12] for s in steps]
        return scale_cache[key]

    phrase = rng.integers(0, 7, size=16)  # motivo repetido (dá "cara" de música)

    for b in range(nbars):
        t0 = b * bar
        root, minor = progression[(b // 2) % len(progression)]
        section = (b // 16) % 4  # 0 intro/leve, 1 cheio, 2 quebra, 3 cheio
        if style == "beats":
            section = 1 if (b // 32) % 4 != 3 else 2  # quebra curta a cada 128 compassos

        # --- bateria
        for q in range(4):
            if section != 2 or q == 0:
                add(drums, K, t0 + q * beat, 0.95)
            if q in (1, 3) and section in (1, 3):
                add(drums, S, t0 + q * beat, 0.45)
            add(drums, H, t0 + q * beat + beat / 2, 0.35 if section != 0 else 0.2)
            if style == "music" and section in (1, 3) and q == 3 and b % 2 == 1:
                add(drums, HO, t0 + q * beat + beat / 2, 0.25)

        # --- baixo (colcheias na tônica, oitava 2)
        if not (style == "music" and section == 0 and b % 16 < 4):
            f = hz(root, 1 if NOTE[root] > 6 else 2)
            for e in range(8):
                if section == 2 and e % 2:
                    continue
                note = tone(f, beat / 2 / SR * 0.95, SOFT_SAW)
                note *= env(len(note), 0.004 * SR, 0.03 * SR)
                add(bass, note, t0 + e * beat / 2, 0.42)

        # --- pad (acorde sustentado por 2 compassos)
        if b % 2 == 0:
            dur = 2 * bar / SR
            sig = np.zeros(int(dur * SR), np.float32)
            for nm in chord_notes(root, minor):
                for det in (0.997, 1.003):
                    sig += tone(hz(nm, 4) * det, dur, TRI)
            sig *= env(len(sig), 0.25 * SR, 0.35 * SR) / 6
            add(pad, sig, t0, 0.5 if style == "beats" else 0.75)

        # --- melodia / arpejo
        if style == "music" and section in (1, 3, 2):
            sc = scale(root, minor)
            step = beat / 2 if section != 2 else beat
            count = 8 if section != 2 else 4
            for e in range(count):
                if section != 2 and (e + b) % 7 == 6:
                    continue  # respiros na frase
                deg = int(phrase[(e + (b % 2) * 8) % 16])
                f = hz(sc[deg], 5)
                note = tone(f, step / SR * 0.9, SQUARE if section != 2 else TRI)
                note *= env(len(note), 0.005 * SR, 0.06 * SR)
                add(lead, note, t0 + e * step, 0.16)
        elif style == "beats" and section == 1 and b % 4 == 3:
            # pequeno arpejo a cada 4 compassos, para a batida não cansar
            for e, nm in enumerate(chord_notes(root, minor) * 2):
                note = tone(hz(nm, 5), beat / 2 / SR * 0.8, TRI)
                note *= env(len(note), 0.004 * SR, 0.05 * SR)
                add(lead, note, t0 + e * beat / 2, 0.12)

    mix = drums * 0.9 + bass + pad + lead
    # fade in / fade out e limitador suave
    f = int(2.0 * SR)
    mix[:f] *= np.linspace(0, 1, f)
    mix[-f:] *= np.linspace(1, 0, f)
    mix = np.tanh(mix * 1.4)
    mix /= max(1e-6, np.max(np.abs(mix))) / 0.92
    return mix.astype(np.float32)


def write_mp3(samples, path, title, artist, album, genre, track):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((samples * 32767).astype(np.int16).tobytes())
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name, "-ac", "1", "-b:a", "96k",
           "-metadata", f"title={title}", "-metadata", f"artist={artist}",
           "-metadata", f"album={album}", "-metadata", f"genre={genre}",
           "-metadata", f"track={track}", "-id3v2_version", "3", path]
    subprocess.run(cmd, check=True)
    os.remove(tmp.name)


# ---------------------------------------------------------------- catálogos
BEATS = [  # (bpm, título)
    (150, "Run Beat 150"), (160, "Run Beat 160"), (165, "Run Beat 165"), (170, "Run Beat 170"),
    (175, "Run Beat 175"), (180, "Run Beat 180"), (185, "Run Beat 185"), (190, "Run Beat 190"),
    (80, "Ride Beat 80"), (90, "Ride Beat 90"), (100, "Ride Beat 100"),
]
BEAT_PROG = [("A", True), ("F", False), ("C", False), ("G", False)]

MUSIC = [  # (bpm, título, progressão)
    (124, "Morning Miles", [("D", False), ("A", False), ("B", True), ("G", False)]),
    (128, "Long Ride", [("E", True), ("C", False), ("G", False), ("D", False)]),
    (140, "Tempo Run", [("F", True), ("C#", False), ("G#", False), ("D#", False)]),
    (150, "Hill Repeats", [("C", True), ("G#", False), ("D#", False), ("A#", False)]),
    (165, "Race Day", [("G", False), ("D", False), ("E", True), ("C", False)]),
    (172, "Final Kick", [("B", True), ("G", False), ("D", False), ("A", False)]),
]


def build(kind, seconds):
    out_dir = os.path.join(ROOT, kind)
    os.makedirs(out_dir, exist_ok=True)
    catalog = []
    if kind == "beats":
        items = [(bpm, title, BEAT_PROG) for bpm, title in BEATS]
        artist, album, genre = "Smart Run Beats", "Smart Run Beats", "Workout"
    else:
        items = MUSIC
        artist, album, genre = "Smart Workout Music", "Smart Workout Music", "Electronic"
    for i, (bpm, title, prog) in enumerate(items, 1):
        tid = ("b" if kind == "beats" else "m") + f"{bpm:03d}" + ("" if kind == "beats" else f"{i}")
        path = os.path.join(out_dir, tid + ".mp3")
        print(f"  {kind}/{tid}.mp3  {title}  {bpm} BPM ...", flush=True)
        samples = render(bpm, seconds, prog, seed=1000 + i * 17 + bpm, style=kind)
        write_mp3(samples, path, f"{title} ({bpm} BPM)", artist, album, genre, i)
        catalog.append({"id": tid, "title": f"{title} ({bpm} BPM)", "bpm": bpm, "seconds": seconds,
                        "bytes": os.path.getsize(path), "url": f"{BASE_URL}/{kind}/{tid}.mp3"})
    with open(os.path.join(out_dir, "catalog.json"), "w", encoding="utf-8") as f:
        json.dump({"name": artist, "tracks": catalog}, f, indent=2)
    total = sum(c["bytes"] for c in catalog) / 1e6
    print(f"  -> {len(catalog)} faixas, {total:.1f} MB")
    return catalog


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["beats", "music"])
    ap.add_argument("--seconds", type=int, default=300)
    a = ap.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg não encontrado no PATH")
    for kind in (["beats", "music"] if not a.only else [a.only]):
        build(kind, a.seconds if kind == "beats" else max(60, a.seconds - 60))
