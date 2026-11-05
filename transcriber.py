"""Live speech-to-text for your microphone AND computer audio at the same time.

Run:  run.bat (Windows)  or  ./setup.sh then ./run.sh (Linux)   (or: python transcriber.py)
Needs: onnx-asr, faster-whisper, soundcard, pywin32. Runs offline after the first model download.

Each source (mic, and a WASAPI loopback of the default speaker) is captured on its own
thread and cut into phrases on silence. Phrases go to one shared speech model and land in
the transcript labeled [Mic] / [PC]. Models:
  - NVIDIA Parakeet TDT (default) via onnx-asr: int8, CPU only, ~12x real time on 2 threads.
  - Whisper via faster-whisper: CPU int8, or GPU only when "Use GPU" is ticked.
The discrete GPU is never touched unless "Use GPU" is ticked, so it can sleep on battery.
New text goes to its own insertion point (after the last transcribed text), so you can edit
anywhere while it runs; tick "Insert at cursor" to have it typed where the cursor is instead.
Save writes <session name>.md under saved_histories/, ready to paste into an LLM.
"""

import collections
import datetime
import glob
import importlib.util
import os
import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import soundcard as sc

try:
    import pythoncom  # Windows: worker threads need COM initialized for WASAPI
except ImportError:
    pythoncom = None

# A windowed .exe has no console; libraries that print progress bars would crash on None.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

APP_DIR = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
SAVE_DIR = os.path.join(APP_DIR, "saved_histories")
# Bundled files live in the PyInstaller temp dir when frozen, next to the script otherwise.
ICON_PATH = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))), "icon.ico")
SAVE_INSTRUCTION = (
    "Make a Markdown report on what was discussed. What are the next steps?\n"
    "Include a short summary, the key points and decisions, and action items "
    "(with owners and deadlines if mentioned).\n"
    "Lines marked [Mic] are me; lines marked [PC] are the other participants / computer audio."
)

# Language -> [(dropdown label, engine, model id)]. The first entry is the default.
MODELS = {
    "English": [
        ("Parakeet v2 (best)", "parakeet", "nemo-parakeet-tdt-0.6b-v2"),
        ("Whisper tiny.en", "whisper", "tiny.en"),
        ("Whisper base.en", "whisper", "base.en"),
        ("Whisper small.en", "whisper", "small.en"),
    ],
    "Multilingual": [
        ("Parakeet v3 (25 EU langs)", "parakeet", "nemo-parakeet-tdt-0.6b-v3"),
        ("Whisper base", "whisper", "base"),
        ("Whisper small", "whisper", "small"),
        ("Whisper large-v3-turbo", "whisper", "large-v3-turbo"),
    ],
}
DEFAULT_LANGUAGE = "English"

# CPU threads per model: more threads mostly cost power, not latency, at phrase lengths.
PARAKEET_THREADS = 2        # 2 threads were as fast as 4 or 8 in testing
WHISPER_CPU_THREADS = 4

RATE = 16000
BLOCK = RATE // 10          # 100 ms per captured block
NOISE_FACTOR = 3.0          # "loud" = this many times above the running noise floor...
MIC_MIN_THRESHOLD = 0.004   # ...and at least this RMS
PC_MIN_THRESHOLD = 0.005
MIN_LOUD_BLOCKS = 3         # ignore blips shorter than 0.3 s
END_SILENCE_BLOCKS = 6      # 0.6 s of silence ends a phrase
LONG_PHRASE_BLOCKS = 50     # after 5 s, cut at the first 0.2 s pause (text shows up sooner)
MAX_PHRASE_BLOCKS = 100     # hard cut at 10 s
PREROLL_BLOCKS = 3


def _add_cuda_dll_dirs():
    """faster-whisper (CTranslate2) needs cuBLAS/cuDNN for GPU. Reuse the copies shipped
    with PyTorch or the nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels, if installed."""
    dirs = []
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec and torch_spec.origin:
        dirs.append(os.path.join(os.path.dirname(torch_spec.origin), "lib"))
    nvidia_spec = importlib.util.find_spec("nvidia")
    if nvidia_spec and nvidia_spec.submodule_search_locations:
        for base in nvidia_spec.submodule_search_locations:
            dirs += glob.glob(os.path.join(base, "*", "bin"))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs + [os.environ["PATH"]])


def has_speech(clip):
    """Silero VAD (tiny, single-threaded): skips noise-only phrases before the big model runs."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    return bool(get_speech_timestamps(clip, VadOptions()))


class ParakeetEngine:
    """NVIDIA Parakeet TDT 0.6B through onnx-asr, int8 on the CPU."""

    def __init__(self, model_id):
        import onnx_asr
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = PARAKEET_THREADS
        opts.inter_op_num_threads = 1
        # Uses the local cache; downloads (~650 MB) only the first time.
        self.model = onnx_asr.load_model(
            model_id, quantization="int8", providers=["CPUExecutionProvider"], sess_options=opts,
        )
        self.device = "cpu"
        self.model.recognize(np.zeros(RATE, dtype=np.float32))  # warm-up

    def transcribe(self, clip, language):
        return self.model.recognize(clip).strip()


class WhisperEngine:
    """Whisper through faster-whisper. CPU int8 unless use_gpu (then CUDA, CPU as fallback)."""

    def __init__(self, model_id, use_gpu):
        if use_gpu:
            _add_cuda_dll_dirs()
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model

        try:
            path = download_model(model_id, local_files_only=True)
        except Exception:
            path = download_model(model_id)
        options = ([("cuda", "float16")] if use_gpu else []) + [("cpu", "int8")]
        for device, compute_type in options:
            try:
                model = WhisperModel(path, device=device, compute_type=compute_type,
                                     cpu_threads=WHISPER_CPU_THREADS)
                list(model.transcribe(np.zeros(RATE, dtype=np.float32), beam_size=1)[0])
                self.model, self.device = model, device
                return
            except Exception:
                if device == "cpu":
                    raise

    def transcribe(self, clip, language):
        segments, _ = self.model.transcribe(
            clip, language=language, beam_size=1, vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()


def capture_worker(source, device, min_threshold, clips, out, stop_event):
    """Record `device` and put (source, audio) phrases on `clips`."""
    if pythoncom:
        pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
    try:
        preroll = collections.deque(maxlen=PREROLL_BLOCKS)
        phrase, silent, loud_count, noise = [], 0, 0, None
        with device.recorder(samplerate=RATE, channels=1, blocksize=BLOCK) as recorder:
            while not stop_event.is_set():
                block = recorder.record(numframes=BLOCK)[:, 0]
                level = float(np.sqrt(np.mean(block ** 2)))
                noise = level if noise is None else noise
                loud = level > max(min_threshold, noise * NOISE_FACTOR)

                if phrase:
                    phrase.append(block)
                    silent = 0 if loud else silent + 1
                    loud_count += loud
                    end_after = 2 if len(phrase) > LONG_PHRASE_BLOCKS else END_SILENCE_BLOCKS
                    if silent >= end_after or len(phrase) >= MAX_PHRASE_BLOCKS:
                        if loud_count >= MIN_LOUD_BLOCKS:
                            clips.put((source, np.concatenate(phrase)))
                            out.put(("partial", source, "transcribing"))
                        else:
                            out.put(("partial", source, None))
                        phrase = []
                elif loud:
                    out.put(("partial", source, "hearing speech"))
                    phrase, silent, loud_count = [*preroll, block], 0, 1
                    preroll.clear()
                else:
                    noise = 0.95 * noise + 0.05 * level
                    preroll.append(block)
        if phrase and loud_count >= MIN_LOUD_BLOCKS:
            clips.put((source, np.concatenate(phrase)))
    except Exception as e:
        out.put(("error", source, str(e)))
    finally:
        if pythoncom:
            pythoncom.CoUninitialize()
        out.put(("stopped", source, None))


def transcribe_worker(spec, use_gpu, models, clips, out):
    """Single consumer: transcribes phrases from all sources in arrival order.
    `models` caches the loaded speech model (only one, to keep RAM low)."""
    label, engine, model_id, language = spec
    key = (engine, model_id, use_gpu and engine == "whisper")
    try:
        if models.get("asr", (None,))[0] != key:
            models.pop("asr", None)  # free the previous model before loading the next
            out.put(("status", "Model", f"loading {label}... (the first use downloads it)"))
            asr = ParakeetEngine(model_id) if engine == "parakeet" else WhisperEngine(model_id, use_gpu)
            models["asr"] = (key, asr)
        asr = models["asr"][1]
        extra = " (GPU unavailable)" if use_gpu and asr.device == "cpu" else ""
        out.put(("status", "Model", f"{label} on {asr.device.upper()}{extra} - listening"))

        whisper_language = "en" if language == "English" else None  # None = auto-detect
        while (item := clips.get()) is not None:
            source, clip = item
            text = asr.transcribe(clip, whisper_language) if has_speech(clip) else ""
            out.put(("final", source, text) if text else ("partial", source, None))
    except Exception as e:
        out.put(("error", "Model", str(e)))
    finally:
        out.put(("stopped", "Model", None))


class App:
    def __init__(self, root):
        self.root = root
        self.queue = queue.Queue()
        self.clips = None
        self.stop_event = threading.Event()
        self.workers = []
        self.running = set()
        self.models = {}
        self.last_source = None
        self.partials = {}
        self.saved_path = None  # file this session last saved to (safe to overwrite)

        root.title("Speech Transcriber")
        root.geometry("820x560")
        try:
            root.iconbitmap(default=ICON_PATH)
        except tk.TclError:
            pass  # .ico unsupported (non-Windows) or missing: keep the default icon

        top_bar = ttk.Frame(root, padding=(8, 8, 8, 0))
        top_bar.pack(fill="x")
        top = ttk.Frame(top_bar)
        top.pack(fill="x")
        model_row = ttk.Frame(top_bar, padding=(0, 6, 0, 0))
        model_row.pack(fill="x")
        session_row = ttk.Frame(top_bar, padding=(0, 6, 0, 0))
        session_row.pack(fill="x")

        self.use_mic = tk.BooleanVar(value=True)
        self.use_pc = tk.BooleanVar(value=True)
        self.mic_check = ttk.Checkbutton(top, text="Mic:", variable=self.use_mic)
        self.mic_check.pack(side="left")
        self.mics = sc.all_microphones()
        self.mic_box = ttk.Combobox(top, values=[m.name for m in self.mics],
                                    state="readonly", width=30)
        if self.mics:
            default_name = sc.default_microphone().name
            names = [m.name for m in self.mics]
            self.mic_box.current(names.index(default_name) if default_name in names else 0)
        else:
            self.use_mic.set(False)
        self.mic_box.pack(side="left", padx=(2, 10))
        self.pc_check = ttk.Checkbutton(top, text="Computer audio", variable=self.use_pc)
        self.pc_check.pack(side="left", padx=(0, 10))
        self.toggle_btn = ttk.Button(top, text="Start", command=self.toggle)
        self.toggle_btn.pack(side="left")

        ttk.Label(model_row, text="Language:").pack(side="left")
        self.language_box = ttk.Combobox(model_row, values=list(MODELS), state="readonly", width=12)
        self.language_box.set(DEFAULT_LANGUAGE)
        self.language_box.pack(side="left", padx=(2, 10))
        self.language_box.bind("<<ComboboxSelected>>", lambda e: self.on_language())
        ttk.Label(model_row, text="Model:").pack(side="left")
        self.model_box = ttk.Combobox(model_row, state="readonly", width=24)
        self.model_box.pack(side="left", padx=(2, 10))
        self.model_box.bind("<<ComboboxSelected>>", lambda e: self.on_model())
        self.use_gpu = tk.BooleanVar(value=False)
        self.gpu_check = ttk.Checkbutton(model_row, text="Use GPU (Whisper only)", variable=self.use_gpu)
        self.gpu_check.pack(side="left")
        self.on_language()

        ttk.Label(session_row, text="Session:").pack(side="left")
        self.session_var = tk.StringVar(value=self.default_session_name())
        ttk.Entry(session_row, textvariable=self.session_var, width=40).pack(side="left", padx=(4, 10))
        ttk.Button(session_row, text="Save", command=self.save).pack(side="left")
        ttk.Button(session_row, text="Copy All", command=self.copy_all).pack(side="left", padx=4)
        ttk.Button(session_row, text="Clear", command=self.clear).pack(side="left")
        # Off (default): new text continues after the last transcribed text, wherever you edit.
        self.at_cursor = tk.BooleanVar(value=False)
        ttk.Checkbutton(session_row, text="Insert at cursor",
                        variable=self.at_cursor).pack(side="left", padx=(10, 0))
        root.bind("<Control-s>", lambda e: self.save())

        text_frame = ttk.Frame(root, padding=(8, 8, 8, 0))
        text_frame.pack(fill="both", expand=True)
        self.text = tk.Text(text_frame, wrap="word", font=("Segoe UI", 12), undo=True)
        scroll = ttk.Scrollbar(text_frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        # Where the next transcribed text goes. Left gravity: anything you type right at this
        # spot lands after it, so your notes never get split by or mixed into new transcript.
        self.text.mark_set("transcript", "end-1c")
        self.text.mark_gravity("transcript", "left")

        self.partial_var = tk.StringVar()
        ttk.Label(root, textvariable=self.partial_var, foreground="gray",
                  padding=(8, 4)).pack(fill="x")
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(root, textvariable=self.status_var, relief="sunken",
                  padding=(8, 2)).pack(fill="x", side="bottom")

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.poll_queue()

    def selected_model(self):
        """-> (label, engine, model id, language) for the current dropdowns."""
        language = self.language_box.get()
        label, engine, model_id = MODELS[language][max(self.model_box.current(), 0)]
        return label, engine, model_id, language

    def on_language(self):
        self.model_box.config(values=[m[0] for m in MODELS[self.language_box.get()]])
        self.model_box.current(0)
        self.on_model()

    def on_model(self):
        # Parakeet is CPU-only by design; the GPU box only applies to Whisper.
        is_whisper = self.selected_model()[1] == "whisper"
        self.gpu_check.config(state="normal" if is_whisper and not self.running else "disabled")

    def set_controls_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.mic_check.config(state=state)
        self.pc_check.config(state=state)
        for box in (self.mic_box, self.language_box, self.model_box):
            box.config(state="readonly" if enabled else "disabled")
        self.on_model()

    def toggle(self):
        if self.running:
            self.stop_event.set()
            self.toggle_btn.config(state="disabled")
            self.status_var.set("Stopping...")
            return

        sources = []
        if self.use_mic.get() and self.mics:
            sources.append(("Mic", self.mics[self.mic_box.current()], MIC_MIN_THRESHOLD))
        if self.use_pc.get():
            speaker = sc.default_speaker()
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            sources.append(("PC", loopback, PC_MIN_THRESHOLD))
        if not sources:
            self.status_var.set("Select at least one source")
            return

        self.stop_event.clear()
        self.clips = queue.Queue()
        spec = self.selected_model()
        use_gpu = self.use_gpu.get() and spec[1] == "whisper"
        self.workers = [threading.Thread(
            target=transcribe_worker,
            args=(spec, use_gpu, self.models, self.clips, self.queue), daemon=True)]
        self.running = {"Model"}
        for name, device, threshold in sources:
            self.workers.append(threading.Thread(
                target=capture_worker,
                args=(name, device, threshold, self.clips, self.queue, self.stop_event),
                daemon=True))
            self.running.add(name)
        for worker in self.workers:
            worker.start()
        self.toggle_btn.config(text="Stop")
        self.set_controls_enabled(False)

    def poll_queue(self):
        try:
            while True:
                kind, source, payload = self.queue.get_nowait()
                if kind == "partial":
                    if payload:
                        self.partials[source] = payload
                    else:
                        self.partials.pop(source, None)
                elif kind == "final":
                    self.partials.pop(source, None)
                    self.append(source, payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "error":
                    self.status_var.set(f"{source} error: {payload}")
                    self.stop_event.set()
                elif kind == "stopped":
                    self.running.discard(source)
                    if source != "Model" and self.running == {"Model"}:
                        self.clips.put(None)  # captures done; let the transcriber drain and exit
                    if not self.running:
                        self.partials.clear()
                        self.toggle_btn.config(text="Start", state="normal")
                        self.set_controls_enabled(True)
                        if "error" not in self.status_var.get():
                            self.status_var.set("Idle")
                self.partial_var.set("   ".join(f"[{s}] {t}..." for s, t in self.partials.items()))
        except queue.Empty:
            pass
        self.root.after(50, self.poll_queue)

    def append(self, source, text):
        """Insert at the transcript mark (default) or the cursor, without moving the cursor
        or scrolling away from what you are editing."""
        at = self.text.index("insert" if self.at_cursor.get() else "transcript")
        before = self.text.get(f"{at} linestart", at)  # what's already on that line
        if source == self.last_source and before.strip():
            chunk = " " + text  # same source keeps going on the same line
        else:
            chunk = ("\n" if before else "") + f"[{source}] {text}"
        # Your own text right after this spot: keep it on its own line.
        tail = "\n" if self.text.get(at, f"{at} lineend").strip() else ""
        follow = self.text.yview()[1] >= 0.999 or self.at_cursor.get()  # was the view at the bottom?
        self.text.edit_separator()  # keep transcript inserts out of your own undo steps
        self.text.insert(at, chunk + tail)
        self.text.edit_separator()
        self.text.mark_set("transcript", f"{at}+{len(chunk)}c")  # right after the new words
        if follow:
            self.text.see("transcript")
        self.last_source = source

    def copy_all(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.report_markdown())
        self.status_var.set("Instruction + transcript copied to clipboard")

    @staticmethod
    def default_session_name():
        return datetime.datetime.now().strftime("session_%Y-%m-%d_%H%M")

    def save(self):
        title = self.session_var.get().strip()
        if not title:
            title = self.default_session_name()
            self.session_var.set(title)
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(" .") or self.default_session_name()
        path = os.path.join(SAVE_DIR, name + ".md")
        if (os.path.exists(path) and path != self.saved_path and not messagebox.askyesno(
                "Overwrite?", f"'{name}.md' already exists in saved_histories. Overwrite it?")):
            return

        saved_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\nSaved: {saved_at}\n\n{self.report_markdown()}")
        self.saved_path = path
        self.status_var.set(f"Saved to {path}")

    def report_markdown(self):
        """Instruction + raw transcript, ready to paste into an LLM."""
        transcript = self.text.get("1.0", "end-1c").strip()
        return f"## Instruction:\n\n{SAVE_INSTRUCTION}\n\n## RAW:\n\n{transcript}\n"

    def clear(self):
        self.text.delete("1.0", "end")
        self.text.mark_set("transcript", "end-1c")
        self.last_source = None
        self.saved_path = None  # new session: don't silently overwrite the previous file
        self.session_var.set(self.default_session_name())

    def on_close(self):
        self.stop_event.set()
        for worker in self.workers[1:]:
            worker.join(timeout=2)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
