from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.image_model import ImageModel
from core.processor import ImageProcessor


class ImageEditorApp:
    """
    GUI controller class (Tkinter).
    Interacts with ImageModel + ImageProcessor (class interaction).
    """

    SUPPORTED_TYPES = [("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("HIT137 - Image Editor (Tkinter + OpenCV)")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

        # State
        self.model = ImageModel()
        self._current_path: str = ""
        self._tk_img: ImageTk.PhotoImage | None = None

        # Slider preview state (prevents 50 undo steps while dragging)
        self._preview_active = False
        self._preview_base: np.ndarray | None = None
        self._preview_effect: str = ""  # "blur" / "edge" / "brightness" / "contrast" / "resize"

        self._build_ui()

    # ---------- UI BUILD ----------
    def _build_ui(self) -> None:
        self._build_menu()

        # Main layout
        container = ttk.Frame(self.root, padding=8)
        container.pack(fill=tk.BOTH, expand=True)

        container.columnconfigure(0, weight=0)  # control panel
        container.columnconfigure(1, weight=1)  # image panel
        container.rowconfigure(0, weight=1)

        self.controls = ttk.Frame(container, width=250)
        self.controls.grid(row=0, column=0, sticky="nsw", padx=(0, 8))

        self.image_panel = ttk.Frame(container)
        self.image_panel.grid(row=0, column=1, sticky="nsew")

        self._build_controls()
        self._build_image_area()
        self._build_status_bar()

        self._set_status("Ready. Use File → Open to load an image.")

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_image)
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_command(label="Save As", command=self.save_image_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.undo)
        edit_menu.add_command(label="Redo", command=self.redo)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        rotate_menu = tk.Menu(menubar, tearoff=0)
        rotate_menu.add_command(label="Rotate 90°", command=lambda: self.apply_rotate(90))
        rotate_menu.add_command(label="Rotate 180°", command=lambda: self.apply_rotate(180))
        rotate_menu.add_command(label="Rotate 270°", command=lambda: self.apply_rotate(270))
        menubar.add_cascade(label="Rotate", menu=rotate_menu)

        flip_menu = tk.Menu(menubar, tearoff=0)
        flip_menu.add_command(label="Flip Horizontal", command=lambda: self.apply_flip("horizontal"))
        flip_menu.add_command(label="Flip Vertical", command=lambda: self.apply_flip("vertical"))
        menubar.add_cascade(label="Flip", menu=flip_menu)

        self.root.config(menu=menubar)

    def _build_controls(self) -> None:
        ttk.Label(self.controls, text="Controls", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))

        btn_frame = ttk.Frame(self.controls)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(btn_frame, text="Grayscale", command=self.apply_grayscale).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Reset to Original", command=self.reset_original).pack(fill=tk.X, pady=2)

        ttk.Separator(self.controls).pack(fill=tk.X, pady=10)

        # Sliders section (meets "at least one slider" requirement)
        ttk.Label(self.controls, text="Adjustments (Slider Preview)", font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self.slider_label = ttk.Label(self.controls, text="Select an effect below to use the slider.")
        self.slider_label.pack(anchor="w", pady=(6, 2))

        self.effect_var = tk.StringVar(value="blur")
        effects = [("Blur", "blur"), ("Edge", "edge"), ("Brightness", "brightness"), ("Contrast", "contrast"), ("Resize %", "resize")]
        for text, val in effects:
            ttk.Radiobutton(
                self.controls,
                text=text,
                value=val,
                variable=self.effect_var,
                command=self._configure_slider_for_effect
            ).pack(anchor="w")

        self.slider = ttk.Scale(self.controls, from_=1, to=25, orient=tk.HORIZONTAL, command=self._on_slider_change)
        self.slider.pack(fill=tk.X, pady=(6, 2))
        self.slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self._on_slider_release)

        self.slider_value = ttk.Label(self.controls, text="")
        self.slider_value.pack(anchor="w")

        ttk.Separator(self.controls).pack(fill=tk.X, pady=10)

        ttk.Label(self.controls, text="Quick Actions", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Button(self.controls, text="Undo", command=self.undo).pack(fill=tk.X, pady=2)
        ttk.Button(self.controls, text="Redo", command=self.redo).pack(fill=tk.X, pady=2)

        self._configure_slider_for_effect()

    def _build_image_area(self) -> None:
        # Canvas for image display
        self.canvas = tk.Canvas(self.image_panel, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _build_status_bar(self) -> None:
        self.status = ttk.Label(self.root, text="", anchor="w")
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    # ---------- STATUS ----------
    def _set_status(self, msg: str) -> None:
        info = self.model.info
        suffix = ""
        if self.model.has_image():
            suffix = f" | {os.path.basename(info.filename)} | {info.width}x{info.height}px"
        self.status.config(text=msg + suffix)

    # ---------- FILE ACTIONS ----------
    def open_image(self) -> None:
        path = filedialog.askopenfilename(title="Open Image", filetypes=self.SUPPORTED_TYPES)
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror("Error", "Could not read that image file.")
            return

        self._current_path = path
        self.model.load_image(bgr, filename=path)
        self._render_current_image()
        self._set_status("Loaded image.")

    def save_image(self) -> None:
        if not self.model.has_image():
            messagebox.showwarning("No image", "Open an image first.")
            return
        if not self._current_path:
            self.save_image_as()
            return
        img = self.model.get_image_copy()
        assert img is not None
        ok = cv2.imwrite(self._current_path, img)
        if not ok:
            messagebox.showerror("Error", "Failed to save the image.")
            return
        self._set_status("Saved image.")

    def save_image_as(self) -> None:
        if not self.model.has_image():
            messagebox.showwarning("No image", "Open an image first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Image As",
            defaultextension=".png",
            filetypes=self.SUPPORTED_TYPES
        )
        if not path:
            return
        img = self.model.get_image_copy()
        assert img is not None
        ok = cv2.imwrite(path, img)
        if not ok:
            messagebox.showerror("Error", "Failed to save the image.")
            return
        self._current_path = path
        self.model.info.filename = path
        self._set_status("Saved image (Save As).")

    def _on_exit(self) -> None:
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.root.destroy()

    # ---------- EDIT ACTIONS ----------
    def undo(self) -> None:
        if not self.model.undo():
            self._set_status("Nothing to undo.")
            return
        self._render_current_image()
        self._set_status("Undo.")

    def redo(self) -> None:
        if not self.model.redo():
            self._set_status("Nothing to redo.")
            return
        self._render_current_image()
        self._set_status("Redo.")

    def reset_original(self) -> None:
        if not self.model.reset_to_original():
            self._set_status("No original to reset to.")
            return
        self._render_current_image()
        self._set_status("Reset to original.")

    # ---------- FILTERS (BUTTON/MENU) ----------
    def apply_grayscale(self) -> None:
        img = self.model.get_image_copy()
        if img is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        out = ImageProcessor.grayscale(img)
        self.model.set_image(out, push_undo=True)
        self._render_current_image()
        self._set_status("Applied grayscale.")

    def apply_rotate(self, degrees: int) -> None:
        img = self.model.get_image_copy()
        if img is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        out = ImageProcessor.rotate(img, degrees)
        self.model.set_image(out, push_undo=True)
        self._render_current_image()
        self._set_status(f"Rotated {degrees}°.")

    def apply_flip(self, mode: str) -> None:
        img = self.model.get_image_copy()
        if img is None:
            messagebox.showwarning("No image", "Open an image first.")
            return
        out = ImageProcessor.flip(img, mode)
        self.model.set_image(out, push_undo=True)
        self._render_current_image()
        self._set_status(f"Flipped {mode}.")

    # ---------- SLIDER PREVIEW ----------
    def _configure_slider_for_effect(self) -> None:
        eff = self.effect_var.get()
        if eff == "blur":
            self.slider.configure(from_=1, to=25)
            self.slider.set(5)
            self.slider_label.config(text="Blur intensity (odd kernel size)")
        elif eff == "edge":
            self.slider.configure(from_=10, to=150)
            self.slider.set(50)
            self.slider_label.config(text="Edge threshold (Canny)")
        elif eff == "brightness":
            self.slider.configure(from_=-100, to=100)
            self.slider.set(0)
            self.slider_label.config(text="Brightness (beta)")
        elif eff == "contrast":
            self.slider.configure(from_=50, to=300)
            self.slider.set(100)
            self.slider_label.config(text="Contrast (alpha %)")
        elif eff == "resize":
            self.slider.configure(from_=10, to=200)
            self.slider.set(100)
            self.slider_label.config(text="Resize (scale %)")
        self.slider_value.config(text=f"Value: {self.slider.get():.0f}")

    def _on_slider_press(self, _event=None) -> None:
        if not self.model.has_image():
            return
        self._preview_active = True
        base = self.model.get_image_copy()
        self._preview_base = base.copy() if base is not None else None
        self._preview_effect = self.effect_var.get()

    def _on_slider_change(self, _value: str) -> None:
        self.slider_value.config(text=f"Value: {self.slider.get():.0f}")
        if not self._preview_active or self._preview_base is None:
            return
        # Live preview without pushing undo history
        out = self._apply_effect_to(self._preview_base, self._preview_effect, self.slider.get())
        self.model.set_image(out, push_undo=False)
        self._render_current_image()
        self._set_status(f"Preview: {self._preview_effect}")

    def _on_slider_release(self, _event=None) -> None:
        if not self._preview_active or self._preview_base is None:
            self._preview_active = False
            return
        # Commit the final effect to history
        out = self._apply_effect_to(self._preview_base, self._preview_effect, self.slider.get())
        self.model.set_image(out, push_undo=True)
        self._render_current_image()
        self._preview_active = False
        self._preview_base = None
        self._set_status(f"Applied {self._preview_effect}.")

    def _apply_effect_to(self, bgr: np.ndarray, eff: str, val: float) -> np.ndarray:
        if eff == "blur":
            return ImageProcessor.blur(bgr, int(val))
        if eff == "edge":
            return ImageProcessor.edge_detect(bgr, int(val))
        if eff == "brightness":
            return ImageProcessor.adjust_brightness(bgr, int(val))
        if eff == "contrast":
            alpha = float(val) / 100.0
            return ImageProcessor.adjust_contrast(bgr, alpha)
        if eff == "resize":
            return ImageProcessor.resize_scale(bgr, int(val))
        return bgr.copy()

    # ---------- RENDER ----------
    def _render_current_image(self) -> None:
        img = self.model.get_image_copy()
        if img is None:
            self.canvas.delete("all")
            return

        # Fit to canvas while keeping aspect ratio
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        iw, ih = pil.size
        scale = min(canvas_w / iw, canvas_h / ih, 1.0)
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._tk_img, anchor="center")

    def run(self) -> None:
        # Re-render on resize
        self.root.bind("<Configure>", lambda _e: self._render_current_image())
        self.root.mainloop()
