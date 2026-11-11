# garmin-audio

Áudio original (gerado por código) dos apps Connect IQ **Smart Run Beats** e
**Smart Workout Music**, servido pelo GitHub Pages em
https://vitinczo-sudo.github.io/garmin-audio/

| Pasta | Conteúdo |
|---|---|
| `beats/` | 11 batidas de treino em BPM exato (150–190 corrida, 80–100 bike), 5 min cada, MP3 96 kbps mono |
| `music/` | 6 faixas instrumentais (124–172 BPM), 4 min cada |
| `*/catalog.json` | lista de faixas (id, título, BPM, duração, tamanho, URL) |
| `tools/gen_audio.py` | gera todo o áudio do zero (numpy + ffmpeg) |
| `tools/check_audio.py` | confere BPM medido, volume, distorção e metadados |

Licença do áudio: CC0 1.0 (domínio público) — ver `LICENSE`.

Regerar e conferir:

```powershell
python tools\gen_audio.py --seconds 300
python tools\check_audio.py      # deve terminar com "RESULTADO: tudo OK"
```
