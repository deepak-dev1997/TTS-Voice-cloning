# flaskapp.py
import io, base64, random, numpy as np, torch
from flask import Flask, request, send_file, jsonify, render_template_string
from chatterbox.tts import ChatterboxTTS
from scipy.io.wavfile import write as wav_write

DEVICE = "cuda"    

MODEL = None         


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = ChatterboxTTS.from_pretrained(DEVICE)
    return MODEL

def synth(params, wav_path=None):
    """Generate audio, return (sr, np.array)."""
    model = get_model()
    if params["seed"] != 0:
        set_seed(int(params["seed"]))

    sr, audio = model.sr, model.generate(
        params["text"],
        audio_prompt_path=wav_path,
        exaggeration=params["exaggeration"],
        temperature=params["temperature"],
        cfg_weight=params["cfgw"],
        min_p=params["min_p"],
        top_p=params["top_p"],
        repetition_penalty=params["rep_penalty"],
    )
    return sr, audio.squeeze(0).numpy()

# ───────────────────────────────── Flask app ────────────────────────────────────
app = Flask(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Chatterbox-TTS</title>
<style>body{font-family:sans-serif;margin:2rem;} label{display:block;margin:0.5rem 0;}
input[type=range]{width:300px;} textarea{width:500px;} details{margin-top:1rem;}
</style></head><body>
<h2>Chatterbox-TTS demo (Flask)</h2>
<form id="ttsForm">
<label>Text:<br><textarea name="text" rows="5" maxlength="300">
Now let's make my mum's favourite. So three mars bars into the pan...
</textarea></label>

<label>Reference audio: <input type="file" name="ref_wav" accept="audio/*"></label>
<label>Exaggeration: <input type="range" name="exaggeration" min="0.25" max="2" step="0.05" value="0.5"></label>
<label>CFG / Pace: <input type="range" name="cfgw" min="0" max="1" step="0.05" value="0.5"></label>

<details><summary>More options</summary>
<label>Seed (0=random): <input type="number" name="seed" value="0"></label>
<label>Temperature: <input type="range" name="temperature" min="0.05" max="5" step="0.05" value="0.8"></label>
<label>min_p: <input type="range" name="min_p" min="0" max="1" step="0.01" value="0.05"></label>
<label>top_p: <input type="range" name="top_p" min="0" max="1" step="0.01" value="1.0"></label>
<label>Repetition penalty: <input type="range" name="rep_penalty" min="1" max="2" step="0.1" value="1.2"></label>
</details><br>
<button type="submit">Generate</button>
</form>
<h3>Output</h3>
<audio id="player" controls></audio>

<script>
document.getElementById('ttsForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const res = await fetch('/generate', {method:'POST', body:fd});
  if (!res.ok){ alert('Error!'); return; }
  const blob = await res.blob();
  document.getElementById('player').src = URL.createObjectURL(blob);
});
</script>
</body></html>"""

# ───── Routes ─────
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)

@app.route("/generate", methods=["POST"])
def generate_route():
    # Handle multipart from UI OR Curl (form-data)
    params = {
        "text"        : request.form.get("text", ""),
        "exaggeration": float(request.form.get("exaggeration", 0.5)),
        "temperature" : float(request.form.get("temperature", 0.8)),
        "seed"        : int(request.form.get("seed", 0)),
        "cfgw"        : float(request.form.get("cfgw", 0.5)),
        "min_p"       : float(request.form.get("min_p", 0.05)),
        "top_p"       : float(request.form.get("top_p", 1.0)),
        "rep_penalty" : float(request.form.get("rep_penalty", 1.2)),
    }
    ref_file = request.files.get("ref_wav")
    ref_path = ref_file.filename if ref_file else None
    if ref_file: ref_file.save(ref_path)

    sr, audio = synth(params, ref_path)

    buf = io.BytesIO()
    wav_write(buf, sr, (audio * 32767).astype("int16"))
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav")

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True)
    params = {
        "text"        : data["text"],
        "exaggeration": data.get("exaggeration", 0.5),
        "temperature" : data.get("temperature" , 0.8),
        "seed"        : data.get("seed", 0),
        "cfgw"        : data.get("cfgw", 0.5),
        "min_p"       : data.get("min_p", 0.05),
        "top_p"       : data.get("top_p", 1.0),
        "rep_penalty" : data.get("rep_penalty", 1.2),
    }
    # Optional: front-end could send a base64 reference wav
    ref_path = data.get("audio_prompt_path")      # existing file path string
    sr, audio = synth(params, ref_path)

    buf = io.BytesIO()
    wav_write(buf, sr, (audio * 32767).astype("int16"))
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"sample_rate": sr, "audio_base64": audio_b64})

# ───── Main ─────
if __name__ == "__main__":
    app.run(port=5009, threaded=True)
