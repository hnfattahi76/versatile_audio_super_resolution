from __future__ import annotations

"""Simple Windows desktop interface for running AudioSR locally.

This script provides a Tkinter-based GUI that mirrors the command-line
options exposed by ``audiosr`` so Windows users can run super-resolution
without using a terminal. Defaults match the CLI developer settings
(guidance scale 3.5, 50 DDIM steps, seed 42, and the default suffix).
"""
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch

from audiosr import (
    build_model,
    get_time,
    read_list,
    save_wave,
    super_resolution,
    super_resolution_long_audio,
)

DEFAULTS = {
    "model_name": "basic",
    "device": "auto",
    "ddim_steps": 50,
    "guidance_scale": 3.5,
    "seed": 42,
    "suffix": "_AudioSR_Processed_48K",
    "chunk_duration": 15,
    "overlap_duration": 2,
    "sample_rate": 48000,
}


class AudioSRDesktop:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AudioSR for Windows")
        self.root.geometry("720x560")

        self.single_file = tk.StringVar()
        self.list_file = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "output"))
        self.model_name = tk.StringVar(value=DEFAULTS["model_name"])
        self.device = tk.StringVar(value=DEFAULTS["device"])
        self.ddim_steps = tk.StringVar(value=str(DEFAULTS["ddim_steps"]))
        self.guidance_scale = tk.StringVar(value=str(DEFAULTS["guidance_scale"]))
        self.seed = tk.StringVar(value=str(DEFAULTS["seed"]))
        self.suffix = tk.StringVar(value=DEFAULTS["suffix"])
        self.chunking_enabled = tk.BooleanVar(value=False)
        self.chunk_duration = tk.StringVar(value=str(DEFAULTS["chunk_duration"]))
        self.overlap_duration = tk.StringVar(value=str(DEFAULTS["overlap_duration"]))
        self.status_text = tk.StringVar(value="Ready to run AudioSR")
        self._deps_verified = False

        self._build_form()

    def _build_form(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        # Input selection
        input_frame = ttk.LabelFrame(container, text="ورودی (Input)")
        input_frame.pack(fill=tk.X, pady=6)

        self._add_file_picker(
            input_frame,
            label="فایل تک (Single audio)",
            variable=self.single_file,
            command=lambda: self._pick_file(self.single_file, [".wav", ".mp3", ".flac", ".ogg", ".m4a"]),
        )

        self._add_file_picker(
            input_frame,
            label="فهرست فایل ها (Text list)",
            variable=self.list_file,
            command=lambda: self._pick_file(self.list_file, [".txt", ".lst", ".csv", ""]),
        )

        self._add_file_picker(
            input_frame,
            label="مسیر خروجی",  # Output path
            variable=self.output_dir,
            command=lambda: self._pick_directory(self.output_dir),
        )

        # Model and sampling options
        options_frame = ttk.LabelFrame(container, text="تنظیمات مدل (Model settings)")
        options_frame.pack(fill=tk.X, pady=6)

        self._add_option(options_frame, "مدل", self.model_name, ["basic", "speech"])

        ttk.Label(options_frame, text="Device (auto/cpu/cuda/mps)").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(options_frame, textvariable=self.device, width=12).grid(row=1, column=1, sticky=tk.W, pady=4)

        ttk.Label(options_frame, text="DDIM steps").grid(row=0, column=2, sticky=tk.W, pady=4)
        ttk.Entry(options_frame, textvariable=self.ddim_steps, width=8).grid(row=0, column=3, sticky=tk.W, pady=4)

        ttk.Label(options_frame, text="Guidance scale").grid(row=1, column=2, sticky=tk.W, pady=4)
        ttk.Entry(options_frame, textvariable=self.guidance_scale, width=8).grid(row=1, column=3, sticky=tk.W, pady=4)

        ttk.Label(options_frame, text="Seed").grid(row=0, column=4, sticky=tk.W, pady=4)
        ttk.Entry(options_frame, textvariable=self.seed, width=10).grid(row=0, column=5, sticky=tk.W, pady=4)

        ttk.Label(options_frame, text="Suffix").grid(row=1, column=4, sticky=tk.W, pady=4)
        ttk.Entry(options_frame, textvariable=self.suffix, width=20).grid(row=1, column=5, sticky=tk.W, pady=4)

        # Chunking options
        chunk_frame = ttk.LabelFrame(container, text="پردازش فایل بلند (Long audio chunking)")
        chunk_frame.pack(fill=tk.X, pady=6)

        ttk.Checkbutton(
            chunk_frame,
            text="فعال کردن برش قطعه ای (Enable chunking)",
            variable=self.chunking_enabled,
        ).grid(row=0, column=0, sticky=tk.W, pady=4)

        ttk.Label(chunk_frame, text="مدت هر قطعه (ثانیه)").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(chunk_frame, textvariable=self.chunk_duration, width=10).grid(row=1, column=1, sticky=tk.W, pady=4)

        ttk.Label(chunk_frame, text="همپوشانی (ثانیه)").grid(row=1, column=2, sticky=tk.W, pady=4)
        ttk.Entry(chunk_frame, textvariable=self.overlap_duration, width=10).grid(row=1, column=3, sticky=tk.W, pady=4)

        # Status and run button
        status_frame = ttk.Frame(container)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=12)

        ttk.Label(status_frame, textvariable=self.status_text, foreground="#1a5276").pack(anchor=tk.W)

        self.run_button = ttk.Button(status_frame, text="اجرای AudioSR", command=self._run_async)
        self.run_button.pack(anchor=tk.E, pady=6)

        self.log_box = tk.Text(status_frame, height=14, wrap=tk.WORD, state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def _add_file_picker(self, parent, label: str, variable: tk.StringVar, command) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=4)
        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        entry = ttk.Entry(frame, textvariable=variable, width=60)
        entry.pack(side=tk.LEFT, padx=8, expand=True, fill=tk.X)
        ttk.Button(frame, text="انتخاب", command=command).pack(side=tk.LEFT)

    def _add_option(self, parent, label: str, variable: tk.StringVar, options: list[str]) -> None:
        ttk.Label(parent, text=label).grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(parent, textvariable=variable, values=options, width=10, state="readonly").grid(row=0, column=1, sticky=tk.W, pady=4)

    def _pick_file(self, variable: tk.StringVar, extensions: list[str]) -> None:
        file_types = [(f"*{ext}", f"*{ext}") for ext in extensions if ext]
        file_path = filedialog.askopenfilename(filetypes=file_types) if file_types else filedialog.askopenfilename()
        if file_path:
            variable.set(file_path)

    def _pick_directory(self, variable: tk.StringVar) -> None:
        directory = filedialog.askdirectory()
        if directory:
            variable.set(directory)

    def _log(self, message: str) -> None:
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.insert(tk.END, f"{message}\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def _run_async(self) -> None:
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self) -> None:
        self.run_button.configure(state=tk.DISABLED)
        self.status_text.set("در حال آماده سازی...")
        self._log("Loading model and validating inputs...")
        try:
            files = self._collect_inputs()
            settings = self._collect_settings()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            self.status_text.set("خطا در ورودی")
            self.run_button.configure(state=tk.NORMAL)
            return

        if not self._deps_verified:
            self.status_text.set("در حال بررسی و نصب پیش‌نیازها...")
            self._log("Ensuring Python dependencies are installed (pip install -r requirements.txt)...")
            try:
                self._ensure_dependencies()
                self._deps_verified = True
            except Exception as exc:  # pragma: no cover - runtime safety
                messagebox.showerror("Dependency error", f"Failed to install dependencies: {exc}")
                self.status_text.set("نصب پیش‌نیازها ناموفق بود")
                self.run_button.configure(state=tk.NORMAL)
                return

        output_dir = Path(self.output_dir.get()).expanduser()
        timestamped_dir = output_dir / get_time()
        timestamped_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Outputs will be written to: {timestamped_dir}")

        torch.set_float32_matmul_precision("high")

        try:
            self.status_text.set("در حال بارگذاری مدل...")
            model = build_model(model_name=settings["model_name"], device=settings["device"])
        except Exception as exc:  # pragma: no cover - runtime safety
            messagebox.showerror("Model error", f"Failed to load AudioSR: {exc}")
            self.status_text.set("بارگذاری مدل ناموفق بود")
            self.run_button.configure(state=tk.NORMAL)
            return

        successes = 0
        failures = []

        for file_path in files:
            name = Path(file_path).stem + settings["suffix"]
            self.status_text.set(f"در حال پردازش: {Path(file_path).name}")
            self._log(f"Processing {file_path} ...")
            try:
                if settings["chunking"]:
                    waveform = super_resolution_long_audio(
                        model,
                        file_path,
                        seed=settings["seed"],
                        guidance_scale=settings["guidance_scale"],
                        ddim_steps=settings["ddim_steps"],
                        chunk_duration_s=settings["chunk_duration"],
                        overlap_duration_s=settings["overlap_duration"],
                    )
                else:
                    waveform = super_resolution(
                        model,
                        file_path,
                        seed=settings["seed"],
                        guidance_scale=settings["guidance_scale"],
                        ddim_steps=settings["ddim_steps"],
                        latent_t_per_second=12.8,
                    )

                save_wave(
                    waveform,
                    inputpath=file_path,
                    savepath=str(timestamped_dir),
                    name=name,
                    samplerate=DEFAULTS["sample_rate"],
                )
                successes += 1
            except Exception as exc:  # pragma: no cover - runtime safety
                failures.append((file_path, str(exc)))
                self._log(f"Failed on {file_path}: {exc}")

        summary = f"تمام شد! {successes} فایل با موفقیت پردازش شد." if not failures else (
            f"Processing finished with {successes} successes and {len(failures)} failures."
        )
        self.status_text.set(summary)

        if failures:
            failure_text = "\n".join([f"{path}: {err}" for path, err in failures])
            messagebox.showwarning("Completed with issues", failure_text)
        else:
            messagebox.showinfo("Completed", summary)

        self.run_button.configure(state=tk.NORMAL)

    def _collect_inputs(self) -> list[str]:
        files: list[str] = []
        if self.single_file.get():
            files.append(self.single_file.get())
        if self.list_file.get():
            files.extend(read_list(self.list_file.get()))

        if not files:
            raise ValueError("حداقل یک فایل ورودی یا لیست فایل لازم است.")

        for file_path in files:
            if not Path(file_path).exists():
                raise ValueError(f"فایل یافت نشد: {file_path}")
        return files

    def _collect_settings(self) -> dict:
        try:
            ddim_steps = int(self.ddim_steps.get())
            guidance_scale = float(self.guidance_scale.get())
            seed = int(self.seed.get())
            chunk_duration = float(self.chunk_duration.get())
            overlap_duration = float(self.overlap_duration.get())
        except ValueError as exc:
            raise ValueError("اعداد وارد شده معتبر نیستند.") from exc

        if self.chunking_enabled.get() and chunk_duration <= overlap_duration:
            raise ValueError("مدت قطعه باید بزرگ تر از همپوشانی باشد.")

        return {
            "model_name": self.model_name.get(),
            "device": self.device.get() or DEFAULTS["device"],
            "ddim_steps": ddim_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "suffix": self.suffix.get() or DEFAULTS["suffix"],
            "chunking": self.chunking_enabled.get(),
            "chunk_duration": chunk_duration,
            "overlap_duration": overlap_duration,
        }

    def _ensure_dependencies(self) -> None:
        requirements_file = Path(__file__).resolve().parent / "requirements.txt"
        if not requirements_file.exists():
            raise FileNotFoundError(f"requirements.txt not found at {requirements_file}")

        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        self._log(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self._log(result.stdout)
            self._log(result.stderr)
            raise RuntimeError(result.stderr.strip() or "Dependency installation failed")
        self._log("Dependencies are up to date.")


def main() -> None:
    root = tk.Tk()
    app = AudioSRDesktop(root)
    root.mainloop()


if __name__ == "__main__":
    main()
