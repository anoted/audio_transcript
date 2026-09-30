"""Live speech-to-text for your microphone AND computer audio at the same time.

Run:  ./setup.sh  then  ./run.sh   (or: python transcriber.py)
Needs: faster-whisper, soundcard, pywin32. Runs offline after the first model download.

Each source (mic, and a WASAPI loopback of the default speaker) is captured on its own
thread and cut into phrases on silence. Phrases go to one shared faster-whisper model
(GPU if available, else CPU) and land in the transcript labeled [Mic] / [PC].
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


_add_cuda_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402
from faster_whisper.utils import download_model  # noqa: E402

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

MODELS = ["tiny.en", "base.en", "small.en"]
DEFAULT_MODEL = "base.en"

RATE = 16000
BLOCK = RATE // 10          # 100 ms per captured block
NOISE_FACTOR = 3.0          # "loud" = this many times above the running noise floor...
MIC_MIN_THRESHOLD = 0.004   # ...and at least this RMS
PC_MIN_THRESHOLD = 0.005
MIN_LOUD_BLOCKS = 3         # ignore blips shorter than 0.3 s
END_SILENCE_BLOCKS = 6      # 0.6 s of silence ends a phrase
LONG_PHRASE_BLOCKS = 80     # after 8 s, cut at the first 0.2 s pause
MAX_PHRASE_BLOCKS = 150     # hard cut at 15 s
PREROLL_BLOCKS = 3


def load_whisper(name):
    """Load from the local cache (no network), downloading only the first time.
    Tries GPU first and verifies it with a tiny warm-up, then falls back to CPU."""
    try:
        path = download_model(name, local_files_only=True)
    except Exception:
        path = download_model(name)
    for device, compute_type in (("cuda", "float16"), ("cpu", "int8")):
        try:
            model = WhisperModel(path, device=device, compute_type=compute_type)
            list(model.transcribe(np.zeros(RATE, dtype=np.float32), beam_size=1)[0])
            return model, device
        except Exception:
            if device == "cpu":
                raise


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


def transcribe_worker(model_name, models, clips, out):
    """Single consumer: transcribes phrases from all sources in arrival order."""
    try:
        if model_name not in models:
            out.put(("status", "Model", f"loading {model_name}..."))
            models[model_name] = load_whisper(model_name)
        model, device = models[model_name]
        out.put(("status", "Model", f"{model_name} on {device.upper()} - listening"))

        while (item := clips.get()) is not None:
            source, clip = item
            segments, _ = model.transcribe(
                clip, language="en", beam_size=1, vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6)
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
        root.geometry("820x520")
        try:
            root.iconbitmap(default=ICON_PATH)
        except tk.TclError:
            pass  # .ico unsupported (non-Windows) or missing: keep the default icon

        top_bar = ttk.Frame(root, padding=(8, 8, 8, 0))
        top_bar.pack(fill="x")
        top = ttk.Frame(top_bar)
        top.pack(fill="x")
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

        ttk.Label(top, text="Model:").pack(side="left")
        self.model_box = ttk.Combobox(top, values=MODELS, state="readonly", width=9)
        self.model_box.set(DEFAULT_MODEL)
        self.model_box.pack(side="left", padx=(2, 10))

        self.toggle_btn = ttk.Button(top, text="Start", command=self.toggle)
        self.toggle_btn.pack(side="left")

        ttk.Label(session_row, text="Session:").pack(side="left")
        self.session_var = tk.StringVar(value=self.default_session_name())
        ttk.Entry(session_row, textvariable=self.session_var, width=40).pack(side="left", padx=(4, 10))
        ttk.Button(session_row, text="Save", command=self.save).pack(side="left")
        ttk.Button(session_row, text="Copy All", command=self.copy_all).pack(side="left", padx=4)
        ttk.Button(session_row, text="Clear", command=self.clear).pack(side="left")
        root.bind("<Control-s>", lambda e: self.save())

        text_frame = ttk.Frame(root, padding=(8, 8, 8, 0))
        text_frame.pack(fill="both", expand=True)
        self.text = tk.Text(text_frame, wrap="word", font=("Segoe UI", 12), undo=True)
        scroll = ttk.Scrollbar(text_frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        self.partial_var = tk.StringVar()
        ttk.Label(root, textvariable=self.partial_var, foreground="gray",
                  padding=(8, 4)).pack(fill="x")
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(root, textvariable=self.status_var, relief="sunken",
                  padding=(8, 2)).pack(fill="x", side="bottom")

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.poll_queue()

    def set_controls_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.mic_check.config(state=state)
        self.pc_check.config(state=state)
        self.mic_box.config(state="readonly" if enabled else "disabled")
        self.model_box.config(state="readonly" if enabled else "disabled")

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
        self.workers = [threading.Thread(
            target=transcribe_worker,
            args=(self.model_box.get(), self.models, self.clips, self.queue), daemon=True)]
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
        existing = self.text.get("1.0", "end-1c")
        if source == self.last_source and existing and not existing.endswith("\n"):
            chunk = " " + text  # same source keeps going on the same line
        else:
            chunk = ("\n" if existing and not existing.endswith("\n") else "") + f"[{source}] {text}"
        self.text.insert("end", chunk)
        self.text.see("end")
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
