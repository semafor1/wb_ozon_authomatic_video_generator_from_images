#!/usr/bin/env python3
import os
import queue
import re
import shutil
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

MARKET_PROFILES = {
    "wildberries": {"size": (1080, 1920)},
    "ozon": {"size": (1080, 1440)},
}

TRANSITIONS = {
    "Push left": "smoothleft",
    "Cross Fade": "fade",
    "Swipe left": "wipeleft",
    "Slide left": "slideleft",
    "Spin Blur": "hblur",
    "Shake": "distance",
    "Glitch": "pixelize",
}

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
MIN_DURATION_SECONDS = 8
TRANSITION_DURATION_SECONDS = 0.5
DEFAULT_CLIP_DURATION_SECONDS = 2.2


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def collect_images(folder_path: str):
    folder = Path(folder_path)
    images = [
        str(path)
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=natural_key)
    return images


def parse_folders(raw_value: str):
    lines = raw_value.replace(";", "\n").splitlines()
    result = []
    seen = set()
    for line in lines:
        path = line.strip().strip('"')
        if not path:
            continue
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        result.append(norm)
    return result


def select_images_for_duration(
    images: list[str], max_duration_seconds: int, min_photo_seconds: float
):
    if len(images) <= 1:
        return images

    min_clip = max(float(min_photo_seconds), TRANSITION_DURATION_SECONDS + 0.001)
    denominator = min_clip - TRANSITION_DURATION_SECONDS
    max_images_allowed = int(
        (float(max_duration_seconds) - TRANSITION_DURATION_SECONDS) / denominator
    )
    max_images_allowed = max(1, max_images_allowed)
    if len(images) <= max_images_allowed:
        return images
    if max_images_allowed == 1:
        return [images[0]]

    last_index = len(images) - 1
    step = last_index / (max_images_allowed - 1)
    indexes = [int(round(i * step)) for i in range(max_images_allowed)]

    for index in range(1, len(indexes)):
        if indexes[index] <= indexes[index - 1]:
            indexes[index] = indexes[index - 1] + 1
    indexes[-1] = last_index
    for index in range(len(indexes) - 2, -1, -1):
        if indexes[index] >= indexes[index + 1]:
            indexes[index] = indexes[index + 1] - 1

    return [images[index] for index in indexes]


def calc_timing(
    images_count: int, max_duration_seconds: int, min_photo_seconds: float
):
    min_clip = max(float(min_photo_seconds), TRANSITION_DURATION_SECONDS + 0.001)

    if images_count <= 1:
        total_duration = max(float(MIN_DURATION_SECONDS), min_clip)
        total_duration = min(total_duration, float(max_duration_seconds))
        clip_duration = total_duration
        return clip_duration, total_duration

    default_clip = max(DEFAULT_CLIP_DURATION_SECONDS, min_clip)
    default_total = (
        images_count * default_clip
        - (images_count - 1) * TRANSITION_DURATION_SECONDS
    )

    if default_total < float(MIN_DURATION_SECONDS):
        total_duration = float(MIN_DURATION_SECONDS)
        clip_duration = (
            total_duration + (images_count - 1) * TRANSITION_DURATION_SECONDS
        ) / images_count
        clip_duration = max(clip_duration, min_clip)
        total_duration = (
            images_count * clip_duration
            - (images_count - 1) * TRANSITION_DURATION_SECONDS
        )
        return clip_duration, total_duration

    if default_total <= float(max_duration_seconds):
        return default_clip, default_total

    clip_duration = (
        float(max_duration_seconds) + (images_count - 1) * TRANSITION_DURATION_SECONDS
    ) / images_count
    clip_duration = max(clip_duration, min_clip)
    total_duration = (
        images_count * clip_duration
        - (images_count - 1) * TRANSITION_DURATION_SECONDS
    )
    return clip_duration, total_duration


def estimate_video_bitrate_kbps(duration_seconds: float):
    if duration_seconds <= 0:
        return 1500
    bits_budget = MAX_FILE_SIZE_BYTES * 8 * 0.92
    bitrate_kbps = int(bits_budget / duration_seconds / 1000)
    return max(700, min(8000, bitrate_kbps))


def build_filter_graph(
    images_count: int,
    width: int,
    height: int,
    clip_duration: float,
    transition_name: str,
    fps: int,
):
    preprocess = []
    chain = []
    for index in range(images_count):
        preprocess.append(
            f"[{index}:v]"
            f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,format=yuv420p[v{index}]"
        )

    if images_count == 1:
        chain.append(f"[v0]fps={fps},format=yuv420p[outv]")
        return ";".join(preprocess + chain)

    current = "v0"
    offset_step = clip_duration - TRANSITION_DURATION_SECONDS
    for index in range(1, images_count):
        output_name = f"x{index}"
        offset = offset_step * index
        chain.append(
            f"[{current}][v{index}]"
            f"xfade=transition={transition_name}:duration={TRANSITION_DURATION_SECONDS:.3f}:"
            f"offset={offset:.3f}[{output_name}]"
        )
        current = output_name

    chain.append(f"[{current}]fps={fps},format=yuv420p[outv]")
    return ";".join(preprocess + chain)


def build_output_path(folder_path: str, market_name: str):
    folder = Path(folder_path)
    stem = folder.name.strip() or "video"
    candidate = folder / f"{stem}_{market_name}.mp4"
    if not candidate.exists():
        return str(candidate)

    counter = 2
    while True:
        candidate = folder / f"{stem}_{market_name}_{counter}.mp4"
        if not candidate.exists():
            return str(candidate)
        counter += 1


def run_ffmpeg_for_folder(
    folder_path: str,
    images: list[str],
    output_path: str,
    width: int,
    height: int,
    transition_name: str,
    max_duration_seconds: int,
    min_photo_seconds: float,
    fps: int,
):
    clip_duration, total_duration = calc_timing(
        len(images), max_duration_seconds, min_photo_seconds
    )
    bitrate_kbps = estimate_video_bitrate_kbps(total_duration)

    command = ["ffmpeg", "-y"]
    for image_path in images:
        command.extend(["-loop", "1", "-t", f"{clip_duration:.3f}", "-i", image_path])

    filter_graph = build_filter_graph(
        images_count=len(images),
        width=width,
        height=height,
        clip_duration=clip_duration,
        transition_name=transition_name,
        fps=fps,
    )

    command.extend(
        [
            "-filter_complex",
            filter_graph,
            "-map",
            "[outv]",
            "-an",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-b:v",
            f"{bitrate_kbps}k",
            "-maxrate",
            f"{bitrate_kbps}k",
            "-bufsize",
            f"{bitrate_kbps * 2}k",
            "-movflags",
            "+faststart",
            output_path,
        ]
    )

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode == 0:
        return True, total_duration, ""
    return False, total_duration, process.stderr.strip()


class VideoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WB / Ozon: видео из фото")
        self.geometry("860x700")
        self.minsize(760, 620)

        self.log_queue = queue.Queue()
        self.worker_thread = None

        self.market_var = tk.StringVar(value="wildberries и ozon")
        self.transition_var = tk.StringVar(value="Slide left")
        self.max_duration_var = tk.IntVar(value=10)
        self.min_photo_duration_var = tk.DoubleVar(value=1.5)
        self.fps_var = tk.IntVar(value=30)
        self.status_var = tk.StringVar(value="Готово к запуску")

        self._build_ui()
        self.after(120, self._flush_queue)

    def _build_ui(self):
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)

        options_frame = ttk.LabelFrame(root, text="Параметры видео")
        options_frame.pack(fill="x")
        options_frame.columnconfigure(0, weight=0)
        options_frame.columnconfigure(1, weight=1)

        ttk.Label(options_frame, text="Маркетплейс:").grid(
            row=0, column=0, padx=8, pady=(8, 4), sticky="w"
        )
        market_combo = ttk.Combobox(
            options_frame,
            textvariable=self.market_var,
            state="readonly",
            values=["wildberries", "ozon", "wildberries и ozon"],
            width=22,
        )
        market_combo.grid(row=0, column=1, padx=8, pady=(8, 4), sticky="ew")

        ttk.Label(options_frame, text="Переходы:").grid(
            row=1, column=0, padx=8, pady=4, sticky="w"
        )
        transition_combo = ttk.Combobox(
            options_frame,
            textvariable=self.transition_var,
            state="readonly",
            values=list(TRANSITIONS.keys()),
            width=18,
        )
        transition_combo.grid(row=1, column=1, padx=8, pady=4, sticky="ew")

        ttk.Label(options_frame, text="FPS (25-30):").grid(
            row=2, column=0, padx=8, pady=4, sticky="w"
        )
        fps_combo = ttk.Combobox(
            options_frame,
            textvariable=self.fps_var,
            state="readonly",
            values=[25, 30],
            width=8,
        )
        fps_combo.grid(row=2, column=1, padx=8, pady=4, sticky="w")

        ttk.Label(options_frame, text="Макс. длительность (сек):").grid(
            row=3, column=0, padx=8, pady=4, sticky="w"
        )
        max_duration_box = ttk.Frame(options_frame)
        max_duration_box.grid(row=3, column=1, padx=8, pady=4, sticky="ew")
        max_duration_box.columnconfigure(0, weight=1)
        duration_scale = ttk.Scale(
            max_duration_box,
            from_=10,
            to=180,
            orient="horizontal",
            variable=self.max_duration_var,
            command=self._on_scale_changed,
        )
        duration_scale.grid(row=0, column=0, sticky="ew")
        self.duration_label = ttk.Label(max_duration_box, text="10", width=4)
        self.duration_label.grid(row=0, column=1, padx=(6, 0), sticky="w")
        self._on_scale_changed(str(self.max_duration_var.get()))

        ttk.Label(options_frame, text="Мин. время на 1 фото (сек):").grid(
            row=4, column=0, padx=8, pady=4, sticky="w"
        )
        min_photo_box = ttk.Frame(options_frame)
        min_photo_box.grid(row=4, column=1, padx=8, pady=4, sticky="ew")
        min_photo_box.columnconfigure(0, weight=1)
        min_photo_scale = ttk.Scale(
            min_photo_box,
            from_=0.5,
            to=10.0,
            orient="horizontal",
            variable=self.min_photo_duration_var,
            command=self._on_min_photo_scale_changed,
        )
        min_photo_scale.grid(row=0, column=0, sticky="ew")
        self.min_photo_duration_label = ttk.Label(min_photo_box, text="1.5", width=4)
        self.min_photo_duration_label.grid(row=0, column=1, padx=(6, 0), sticky="w")
        self._on_min_photo_scale_changed(str(self.min_photo_duration_var.get()))

        note = (
            "Профили: wildberries = 1080x1920 (9:16), "
            "ozon = 1080x1440 (3:4). "
            "Фото вписываются полностью, пустые зоны заполняются черным. "
            "Если фото слишком много, скрипт равномерно выберет часть фото, "
            "чтобы выдержать минимум секунд на кадр."
        )
        ttk.Label(
            options_frame,
            text=note,
            foreground="#3a3a3a",
            justify="left",
            wraplength=620,
        ).grid(
            row=5, column=0, columnspan=2, padx=8, pady=(6, 8), sticky="w"
        )

        folders_frame = ttk.LabelFrame(root, text="Папки с фотографиями")
        folders_frame.pack(fill="both", expand=False, pady=(12, 0))

        ttk.Label(
            folders_frame,
            text='Вставьте пути построчно или через ";" (пример: C:\\images\\product_01)',
        ).pack(anchor="w", padx=8, pady=(8, 4))

        self.paths_text = ScrolledText(folders_frame, height=8, wrap="none")
        self.paths_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._setup_paths_text_editing()

        buttons_frame = ttk.Frame(folders_frame)
        buttons_frame.pack(fill="x", padx=8, pady=(0, 8))

        ttk.Button(
            buttons_frame, text="Добавить папки (мультивыбор)", command=self._add_folder_dialog
        ).pack(side="left")
        ttk.Button(
            buttons_frame, text="Вставить из буфера", command=self._paste_into_paths
        ).pack(side="left", padx=(8, 0))
        ttk.Button(buttons_frame, text="Очистить поле", command=self._clear_paths).pack(
            side="left", padx=(8, 0)
        )

        run_frame = ttk.Frame(root)
        run_frame.pack(fill="x", pady=(12, 0))

        self.run_button = ttk.Button(
            run_frame, text="Собрать видео", command=self.start_processing
        )
        self.run_button.pack(side="left")

        self.progress = ttk.Progressbar(run_frame, mode="indeterminate", length=220)
        self.progress.pack(side="left", padx=(12, 0))

        ttk.Label(run_frame, textvariable=self.status_var).pack(side="left", padx=(12, 0))

        logs_frame = ttk.LabelFrame(root, text="Лог")
        logs_frame.pack(fill="both", expand=True, pady=(12, 0))

        self.log_text = ScrolledText(logs_frame, height=12, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    def _on_scale_changed(self, _):
        value = int(round(float(self.max_duration_var.get())))
        value = max(10, min(180, value))
        self.max_duration_var.set(value)
        self.duration_label.configure(text=str(value))

    def _on_min_photo_scale_changed(self, _):
        value = round(float(self.min_photo_duration_var.get()) * 10) / 10
        value = max(0.5, min(10.0, value))
        self.min_photo_duration_var.set(value)
        self.min_photo_duration_label.configure(text=f"{value:.1f}")

    def _setup_paths_text_editing(self):
        self.paths_context_menu = tk.Menu(self, tearoff=0)
        self.paths_context_menu.add_command(
            label="Вырезать", command=lambda: self.paths_text.event_generate("<<Cut>>")
        )
        self.paths_context_menu.add_command(
            label="Копировать", command=lambda: self.paths_text.event_generate("<<Copy>>")
        )
        self.paths_context_menu.add_command(
            label="Вставить", command=lambda: self.paths_text.event_generate("<<Paste>>")
        )
        self.paths_context_menu.add_separator()
        self.paths_context_menu.add_command(
            label="Выделить все", command=self._select_all_paths
        )

        self.paths_text.bind(
            "<Control-c>", lambda event: self._dispatch_text_event(event, "<<Copy>>")
        )
        self.paths_text.bind(
            "<Control-v>", lambda event: self._dispatch_text_event(event, "<<Paste>>")
        )
        self.paths_text.bind(
            "<Control-x>", lambda event: self._dispatch_text_event(event, "<<Cut>>")
        )
        self.paths_text.bind("<Control-a>", self._on_paths_select_all)
        self.paths_text.bind("<Control-Insert>", self._on_paths_copy_insert)
        self.paths_text.bind("<Shift-Insert>", self._on_paths_paste_insert)
        self.paths_text.bind("<Button-3>", self._show_paths_context_menu)

    def _dispatch_text_event(self, event, virtual_event: str):
        event.widget.event_generate(virtual_event)
        return "break"

    def _on_paths_copy_insert(self, event):
        return self._dispatch_text_event(event, "<<Copy>>")

    def _on_paths_paste_insert(self, event):
        return self._dispatch_text_event(event, "<<Paste>>")

    def _on_paths_select_all(self, _event):
        self._select_all_paths()
        return "break"

    def _select_all_paths(self):
        self.paths_text.focus_set()
        self.paths_text.tag_add("sel", "1.0", "end-1c")
        self.paths_text.mark_set("insert", "1.0")
        self.paths_text.see("insert")

    def _show_paths_context_menu(self, event):
        self.paths_context_menu.tk_popup(event.x_root, event.y_root)
        self.paths_context_menu.grab_release()

    def _paste_into_paths(self):
        try:
            text = self.clipboard_get()
        except tk.TclError:
            return
        if not text:
            return
        self.paths_text.focus_set()
        self.paths_text.insert("insert", text)

    def _append_paths(self, paths: list[str]):
        existing = parse_folders(self.paths_text.get("1.0", "end"))
        existing_set = {os.path.normpath(path) for path in existing}
        to_add = []

        for raw_path in paths:
            path = raw_path.strip().strip('"')
            if not path:
                continue
            norm = os.path.normpath(path)
            if norm in existing_set:
                continue
            existing_set.add(norm)
            to_add.append(norm)

        if not to_add:
            return 0

        current = self.paths_text.get("1.0", "end").strip()
        payload = "\n".join(to_add)
        if current:
            payload = f"\n{payload}"
        self.paths_text.insert("end", payload)
        return len(to_add)

    def _choose_folders_dialog(self, title: str, candidates: list[str]):
        result = {"paths": []}
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.geometry("900x520")
        dialog.minsize(720, 420)
        dialog.transient(self)
        dialog.grab_set()

        root_frame = ttk.Frame(dialog, padding=12)
        root_frame.pack(fill="both", expand=True)

        ttk.Label(
            root_frame,
            text="Выберите папки с фото (можно отметить несколько кликом мыши):",
        ).pack(anchor="w")

        listbox = tk.Listbox(
            root_frame,
            selectmode="multiple",
            exportselection=False,
            activestyle="none",
        )
        listbox.pack(fill="both", expand=True, pady=(8, 8))

        for path in candidates:
            listbox.insert("end", path)

        buttons = ttk.Frame(root_frame)
        buttons.pack(fill="x")

        def select_all():
            listbox.selection_set(0, "end")

        def clear_selection():
            listbox.selection_clear(0, "end")

        def confirm():
            selected_indexes = listbox.curselection()
            result["paths"] = [candidates[i] for i in selected_indexes]
            dialog.destroy()

        def cancel():
            dialog.destroy()

        ttk.Button(buttons, text="Выбрать все", command=select_all).pack(side="left")
        ttk.Button(buttons, text="Снять выбор", command=clear_selection).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(buttons, text="Отмена", command=cancel).pack(side="right")
        ttk.Button(buttons, text="Добавить выбранные", command=confirm).pack(
            side="right", padx=(0, 8)
        )

        dialog.bind("<Escape>", lambda _: cancel())
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        self.wait_window(dialog)
        return result["paths"]

    def _add_folder_dialog(self):
        root_folder = filedialog.askdirectory(title="Выберите родительскую папку")
        if not root_folder:
            return

        candidates = []
        root_images = collect_images(root_folder)
        if root_images:
            candidates.append(os.path.normpath(root_folder))

        root_path = Path(root_folder)
        subfolders = [path for path in root_path.iterdir() if path.is_dir()]
        subfolders.sort(key=lambda item: natural_key(item.name))

        for folder in subfolders:
            if collect_images(str(folder)):
                candidates.append(os.path.normpath(str(folder)))

        if not candidates:
            candidates = [os.path.normpath(root_folder)]

        if len(candidates) == 1:
            selected_paths = candidates
        else:
            selected_paths = self._choose_folders_dialog(
                title="Выбор папок с фото",
                candidates=candidates,
            )

        added_count = self._append_paths(selected_paths)
        if added_count:
            self.status_var.set(f"Добавлено папок: {added_count}")

    def _clear_paths(self):
        self.paths_text.delete("1.0", "end")

    def _append_log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{text}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _flush_queue(self):
        try:
            while True:
                event, payload = self.log_queue.get_nowait()
                if event == "log":
                    self._append_log(payload)
                elif event == "done":
                    self._on_done(payload)
        except queue.Empty:
            pass
        self.after(120, self._flush_queue)

    def _log(self, text: str):
        self.log_queue.put(("log", text))

    def _selected_markets(self):
        value = self.market_var.get()
        if value == "wildberries и ozon":
            return ["wildberries", "ozon"]
        return [value]

    def start_processing(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("В процессе", "Обработка уже запущена.")
            return

        if shutil.which("ffmpeg") is None:
            messagebox.showerror(
                "ffmpeg не найден",
                "Не найден ffmpeg в PATH. Установите ffmpeg и перезапустите скрипт.",
            )
            return

        folders = parse_folders(self.paths_text.get("1.0", "end"))
        if not folders:
            messagebox.showwarning("Нет папок", "Добавьте хотя бы одну папку.")
            return

        missing = [folder for folder in folders if not os.path.isdir(folder)]
        if missing:
            preview = "\n".join(missing[:4])
            more = "" if len(missing) <= 4 else "\n..."
            messagebox.showerror(
                "Папки не найдены",
                f"Следующие пути не существуют:\n{preview}{more}",
            )
            return

        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Идет обработка...")
        self._append_log("-" * 64)
        self._append_log("Новая задача запущена.")

        args = (
            folders,
            self._selected_markets(),
            TRANSITIONS[self.transition_var.get()],
            int(self.max_duration_var.get()),
            float(self.min_photo_duration_var.get()),
            int(self.fps_var.get()),
        )
        self.worker_thread = threading.Thread(
            target=self._worker, args=args, daemon=True
        )
        self.worker_thread.start()

    def _worker(
        self,
        folders: list[str],
        markets: list[str],
        transition_name: str,
        max_duration_seconds: int,
        min_photo_seconds: float,
        fps: int,
    ):
        ok_count = 0
        fail_count = 0

        self._log(
            f"Папок: {len(folders)} | Режимов: {', '.join(markets)} | "
            f"Переход: {transition_name} | Макс. длительность: {max_duration_seconds} сек | "
            f"Мин. на фото: {min_photo_seconds:.1f} сек | FPS: {fps}"
        )

        for folder_index, folder_path in enumerate(folders, start=1):
            images = collect_images(folder_path)
            if not images:
                fail_count += 1
                self._log(f"[{folder_index}/{len(folders)}] {folder_path} -> фото не найдены")
                continue

            self._log(
                f"[{folder_index}/{len(folders)}] {folder_path} -> найдено фото: {len(images)}"
            )
            selected_images = select_images_for_duration(
                images=images,
                max_duration_seconds=max_duration_seconds,
                min_photo_seconds=min_photo_seconds,
            )
            if len(selected_images) < len(images):
                self._log(
                    f"  -> выбрано фото: {len(selected_images)} из {len(images)} "
                    f"(ограничения: мин. {min_photo_seconds:.1f} сек/фото, "
                    f"макс. {max_duration_seconds} сек на видео)"
                )

            for market_name in markets:
                width, height = MARKET_PROFILES[market_name]["size"]
                output_path = build_output_path(folder_path, market_name)
                self._log(
                    f"  -> {market_name}: {width}x{height}, файл: {output_path}"
                )

                ok, total_duration, error_text = run_ffmpeg_for_folder(
                    folder_path=folder_path,
                    images=selected_images,
                    output_path=output_path,
                    width=width,
                    height=height,
                    transition_name=transition_name,
                    max_duration_seconds=max_duration_seconds,
                    min_photo_seconds=min_photo_seconds,
                    fps=fps,
                )
                if not ok and transition_name != "fade":
                    self._log(
                        "  -> выбранный переход не сработал в текущем ffmpeg, "
                        "пробую fallback: fade"
                    )
                    ok, total_duration, error_text = run_ffmpeg_for_folder(
                        folder_path=folder_path,
                        images=selected_images,
                        output_path=output_path,
                        width=width,
                        height=height,
                        transition_name="fade",
                        max_duration_seconds=max_duration_seconds,
                        min_photo_seconds=min_photo_seconds,
                        fps=fps,
                    )

                if ok:
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    info = (
                        f"  -> готово: {output_path} | "
                        f"длительность ~{total_duration:.1f} сек | размер {size_mb:.2f} МБ"
                    )
                    if size_mb > 50:
                        info += " (внимание: больше 50 МБ)"
                    self._log(info)
                    ok_count += 1
                else:
                    fail_count += 1
                    short_err = error_text[-700:] if error_text else "неизвестная ошибка ffmpeg"
                    self._log(f"  -> ошибка ffmpeg: {short_err}")

        self.log_queue.put(("done", {"ok": ok_count, "fail": fail_count}))

    def _on_done(self, stats: dict):
        self.progress.stop()
        self.run_button.configure(state="normal")
        self.status_var.set("Готово")
        self._append_log(
            f"Завершено. Успешно: {stats['ok']} | С ошибками: {stats['fail']}"
        )
        self._append_log("-" * 64)


def main():
    app = VideoApp()
    app.mainloop()


if __name__ == "__main__":
    main()
