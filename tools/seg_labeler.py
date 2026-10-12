"""Polygon labeler for YOLO-seg label files: review and correct prelabels.

Reads <labels>/<stem>.txt, writes corrected files to <out>/<stem>.txt; the prelabels are never overwritten.
A file in <out> means the image is reviewed; an empty file means "reviewed, no polygons".

  python tools/seg_labeler.py \
    --images normalized/bell_pepper/images \
    --labels prelabel/bell_pepper/plantdoc_prelabels/labels \
    --out prelabel/bell_pepper/plantdoc_corrected \
    --classes prelabel/bell_pepper/plantdoc_prelabels/classes.txt \
    --review prelabel/bell_pepper/plantdoc_prelabels/review.csv

Start from the original PlantDoc boxes instead (detection lines load as rectangles; export them back with --export-boxes):
  python tools/seg_labeler.py --images normalized/bell_pepper/images \
    --labels normalized/bell_pepper/labels/detection/condition --match 'plantdoc_*' \
    --classes normalized/bell_pepper/labels/detection/condition/classes.txt --out prelabel/bell_pepper/plantdoc_boxes_corrected

Mouse: left = select polygon / add point while drawing, right = close polygon, wheel = zoom, middle drag = pan.
Export reviewed polygons as YOLO detection boxes (one box per polygon, same class ids):
  python tools/seg_labeler.py --out prelabel/bell_pepper/plantdoc_corrected \
    --classes prelabel/bell_pepper/plantdoc_prelabels/classes.txt --export-boxes prelabel/bell_pepper/plantdoc_corrected_boxes

Keys:  n new polygon, Enter close polygon, BackSpace undo point, Esc cancel, 1-9 class, Delete/d delete selected,
       Ctrl+Z undo, s save, Right/Left next/prev (unsaved edits are auto-saved), h hide labels, f fit.
"""
import argparse
import copy
import csv
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#7f4fc9", "#0d366b", "#898781", "#d64fa0"]
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def read_yolo_seg(path):
    # -> [[class_id, [(x, y), ...]]] with normalized coordinates
    polys = []
    if path.exists():
        for line in path.read_text().split("\n"):
            v = line.split()
            if len(v) == 5:  # YOLO detection box "cls cx cy w h" -> rectangle polygon (a polygon line has >= 7 values)
                cx, cy, w, h = map(float, v[1:])
                polys.append([int(v[0]), [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]])
            elif v:
                xy = list(map(float, v[1:]))
                polys.append([int(v[0]), list(zip(xy[0::2], xy[1::2]))])
    return polys


def write_yolo_seg(path, polys):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} " + " ".join(f"{min(max(v, 0), 1):.6f}" for p in pts for v in p) for c, pts in polys if len(pts) >= 3]
    path.write_text("".join(l + "\n" for l in lines))


def polygons_to_boxes(polys):
    # YOLO detection lines "cls cx cy w h" from each polygon's bounding box (normalized, clipped to the image)
    lines = []
    for c, pts in polys:
        xs, ys = [min(max(p[0], 0), 1) for p in pts], [min(max(p[1], 0), 1) for p in pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        lines.append(f"{c} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
    return lines


def export_boxes(out, dest, classes):
    # only reviewed images (files in --out); an empty polygon file becomes an empty detection file (verified negative)
    dest.mkdir(parents=True, exist_ok=True)
    files = [p for p in out.glob("*.txt") if p.name != "classes.txt"]
    for p in files:
        (dest / p.name).write_text("".join(l + "\n" for l in polygons_to_boxes(read_yolo_seg(p))))
    (dest / "classes.txt").write_text(classes.read_text())
    print(f"wrote {len(files)} detection files to {dest}")


def inside(x, y, pts):
    # ray casting point-in-polygon
    hit = False
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def area(pts):
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])))


class Labeler:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.classes = Path(args.classes).read_text().split()
        self.out = Path(args.out)
        stems = sorted(p.stem for p in Path(args.labels).glob(f"{args.match}.txt") if p.name != "classes.txt")
        self.flags = {}
        if args.review:  # suspicious images first, in review.csv order
            order = []
            with open(args.review) as fh:
                for row in csv.DictReader(fh):
                    stem = Path(row["image"]).stem
                    if stem not in order:
                        order.append(stem)
                    marks = [k for k in ("no_prediction", "mixed_classes", "low_conf", "tiny", "huge") if row.get(k) == "True"]
                    self.flags.setdefault(stem, set()).update(marks)
            stems = [s for s in order if s in stems] + [s for s in stems if s not in order]
        self.images = {p.stem: p for p in Path(args.images).iterdir() if p.suffix.lower() in EXTS}
        self.stems = [s for s in stems if s in self.images]
        if not self.stems:
            sys.exit("no label file in --labels has a matching image in --images")
        self.index, self.cls, self.selected, self.drawing = 0, 0, None, None
        self.hide, self.dirty, self.undo_stack = False, False, []
        self.build_ui()
        self.load(0)

    # ---------------------------------------------------------------- UI
    def build_ui(self):
        self.root.title("seg labeler")
        bar = tk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X)
        for text, cmd in (("◀ Prev", self.prev), ("Next ▶", self.next), ("Next unreviewed", self.next_unreviewed),
                          ("Save (s)", self.save), ("New polygon (n)", self.start_polygon), ("Close polygon (Enter)", self.close_polygon),
                          ("Delete selected (d)", self.delete_selected), ("Clear all", self.clear_all), ("Undo (Ctrl+Z)", self.undo),
                          ("Hide labels (h)", self.toggle_hide), ("Fit (f)", self.fit)):
            tk.Button(bar, text=text, command=cmd).pack(side=tk.LEFT, padx=1, pady=2)
        cls_bar = tk.Frame(self.root)
        cls_bar.pack(side=tk.TOP, fill=tk.X)
        self.cls_buttons = []
        for i, name in enumerate(self.classes):
            b = tk.Button(cls_bar, text=f"{i + 1}: {name}", fg="white", bg=COLORS[i % len(COLORS)],
                          activebackground=COLORS[i % len(COLORS)], command=lambda i=i: self.set_class(i))
            b.pack(side=tk.LEFT, padx=1, pady=2)
            self.cls_buttons.append(b)
        self.status = tk.Label(self.root, anchor="w", font=("TkDefaultFont", 10))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas = tk.Canvas(self.root, bg="#202020", width=1200, height=800, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        c = self.canvas
        c.bind("<Button-1>", self.on_left)
        c.bind("<Button-3>", lambda e: self.close_polygon())
        c.bind("<Button-2>", self.pan_start)
        c.bind("<B2-Motion>", self.pan_move)
        c.bind("<MouseWheel>", lambda e: self.zoom(e, 1.25 if e.delta > 0 else 0.8))
        c.bind("<Button-4>", lambda e: self.zoom(e, 1.25))
        c.bind("<Button-5>", lambda e: self.zoom(e, 0.8))
        c.bind("<Configure>", lambda e: self.fit())  # fit on window resize
        keys = {"<Right>": self.next, "<Left>": self.prev, "s": self.save, "n": self.start_polygon, "<Return>": self.close_polygon,
                "<BackSpace>": self.undo_point, "<Escape>": self.cancel, "d": self.delete_selected, "<Delete>": self.delete_selected,
                "<Control-z>": self.undo, "h": self.toggle_hide, "f": self.fit}
        for key, fn in keys.items():
            self.root.bind(key, lambda e, fn=fn: fn())
        for i in range(min(9, len(self.classes))):
            self.root.bind(str(i + 1), lambda e, i=i: self.set_class(i))
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    # ---------------------------------------------------------------- data
    def load(self, index):
        self.index = index % len(self.stems)
        stem = self.stems[self.index]
        self.image = ImageOps.exif_transpose(Image.open(self.images[stem])).convert("RGB")
        saved = self.out / f"{stem}.txt"
        self.polys = read_yolo_seg(saved if saved.exists() else Path(self.args.labels) / f"{stem}.txt")
        self.selected, self.drawing, self.dirty, self.undo_stack = None, None, False, []
        self.fit()

    def save(self):
        write_yolo_seg(self.out / f"{self.stems[self.index]}.txt", self.polys)
        self.dirty = False
        self.render()

    def change(self):
        # call before every edit: enables undo and auto-save
        self.undo_stack.append(copy.deepcopy(self.polys))
        self.dirty = True

    def go(self, index):
        if self.dirty:
            self.save()
        self.load(index)

    def next(self):
        self.go(self.index + 1)

    def prev(self):
        self.go(self.index - 1)

    def next_unreviewed(self):
        for step in range(1, len(self.stems) + 1):
            i = (self.index + step) % len(self.stems)
            if not (self.out / f"{self.stems[i]}.txt").exists():
                return self.go(i)
        messagebox.showinfo("seg labeler", "every image is reviewed")

    def quit(self):
        if self.dirty and messagebox.askyesno("seg labeler", "Save changes to this image before quitting?"):
            self.save()
        self.root.destroy()

    # ---------------------------------------------------------------- editing
    def set_class(self, i):
        self.cls = i
        if self.selected is not None and self.polys[self.selected][0] != i:
            self.change()
            self.polys[self.selected][0] = i
        self.render()

    def start_polygon(self):
        self.selected, self.drawing = None, []
        self.render()

    def close_polygon(self):
        if self.drawing is not None and len(self.drawing) >= 3:
            self.change()
            self.polys.append([self.cls, self.drawing])
            self.selected = len(self.polys) - 1
        self.drawing = None
        self.render()

    def undo_point(self):
        if self.drawing:
            self.drawing.pop()
            self.render()

    def cancel(self):
        self.drawing, self.selected = None, None
        self.render()

    def delete_selected(self):
        if self.selected is not None:
            self.change()
            del self.polys[self.selected]
            self.selected = None
            self.render()

    def clear_all(self):
        if self.polys:
            self.change()
            self.polys, self.selected = [], None
            self.render()

    def undo(self):
        if self.undo_stack:
            self.polys, self.selected = self.undo_stack.pop(), None
            self.dirty = True
            self.render()

    def toggle_hide(self):
        self.hide = not self.hide
        self.render()

    # ---------------------------------------------------------------- view: screen = image_px * scale + offset
    def fit(self):
        cw, ch = max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)
        self.scale = min(cw / self.image.width, ch / self.image.height)
        self.ox, self.oy = (cw - self.image.width * self.scale) / 2, (ch - self.image.height * self.scale) / 2
        self.render()

    def to_norm(self, sx, sy):
        return (sx - self.ox) / self.scale / self.image.width, (sy - self.oy) / self.scale / self.image.height

    def to_screen(self, x, y):
        return x * self.image.width * self.scale + self.ox, y * self.image.height * self.scale + self.oy

    def zoom(self, e, factor):
        self.ox, self.oy = e.x - (e.x - self.ox) * factor, e.y - (e.y - self.oy) * factor
        self.scale *= factor
        self.render()

    def pan_start(self, e):
        self.pan = (e.x, e.y)

    def pan_move(self, e):
        self.ox, self.oy = self.ox + e.x - self.pan[0], self.oy + e.y - self.pan[1]
        self.pan = (e.x, e.y)
        self.render()

    def on_left(self, e):
        x, y = self.to_norm(e.x, e.y)
        if self.drawing is not None:
            if 0 <= x <= 1 and 0 <= y <= 1:
                self.drawing.append((x, y))
        else:  # select the smallest polygon under the cursor
            hits = [i for i, (_, pts) in enumerate(self.polys) if inside(x, y, pts)]
            self.selected = min(hits, key=lambda i: area(self.polys[i][1])) if hits else None
            if self.selected is not None:
                self.cls = self.polys[self.selected][0]
        self.render()

    # ---------------------------------------------------------------- drawing
    def render(self):
        c = self.canvas
        c.delete("all")
        cw, ch = c.winfo_width(), c.winfo_height()
        # crop only the visible part of the image, so deep zoom stays fast
        x0, y0 = max(0, -self.ox / self.scale), max(0, -self.oy / self.scale)
        x1, y1 = min(self.image.width, (cw - self.ox) / self.scale), min(self.image.height, (ch - self.oy) / self.scale)
        if x1 > x0 and y1 > y0:
            w, h = max(1, int((x1 - x0) * self.scale)), max(1, int((y1 - y0) * self.scale))
            self.tk_image = ImageTk.PhotoImage(self.image.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1)).resize((w, h)))
            c.create_image(x0 * self.scale + self.ox, y0 * self.scale + self.oy, image=self.tk_image, anchor="nw")
        if not self.hide:
            for i, (cls, pts) in enumerate(self.polys):
                color = COLORS[cls % len(COLORS)]
                flat = [v for p in pts for v in self.to_screen(*p)]
                selected = i == self.selected
                c.create_polygon(flat, outline="white" if selected else color, width=3 if selected else 2,
                                 fill=color if selected else "", stipple="gray25" if selected else "")
        if self.drawing is not None:
            flat = [v for p in self.drawing for v in self.to_screen(*p)]
            color = COLORS[self.cls % len(COLORS)]
            if len(flat) >= 4:
                c.create_line(flat, fill=color, width=2)
            for sx, sy in zip(flat[0::2], flat[1::2]):
                c.create_oval(sx - 3, sy - 3, sx + 3, sy + 3, fill=color, outline="white")
        for i, b in enumerate(self.cls_buttons):
            b.config(relief=tk.SUNKEN if i == self.cls else tk.RAISED, bd=4 if i == self.cls else 1)
        stem = self.stems[self.index]
        reviewed = sum((self.out / f"{s}.txt").exists() for s in self.stems)
        state = "UNSAVED" if self.dirty else ("reviewed" if (self.out / f"{stem}.txt").exists() else "not reviewed")
        mode = f"DRAWING {len(self.drawing)} pts (right click / Enter closes)" if self.drawing is not None else "select"
        flags = ", ".join(sorted(self.flags.get(stem, ()))) or "-"
        self.status.config(text=f"[{self.index + 1}/{len(self.stems)}] {self.images[stem].name}   {state}   "
                                f"polygons: {len(self.polys)}   class: {self.classes[self.cls]}   mode: {mode}   "
                                f"flags: {flags}   reviewed {reviewed}/{len(self.stems)}")


def selftest():
    import tempfile
    square = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]
    assert inside(0.3, 0.3, square) and not inside(0.6, 0.3, square)
    assert abs(area(square) - 0.16) < 1e-9
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.txt"
        write_yolo_seg(path, [[2, square], [0, [(0, 0), (1, 0)]]])  # the 2-point polygon must be dropped
        assert read_yolo_seg(path) == [[2, square]]
        write_yolo_seg(path, [])
        assert path.read_text() == "" and read_yolo_seg(path) == []
        path.write_text("1 0.300000 0.300000 0.400000 0.400000\n")  # a detection box loads as a rectangle
        (c, pts), = read_yolo_seg(path)
        assert c == 1 and all(abs(a - b) < 1e-9 for p, q in zip(pts, square) for a, b in zip(p, q))
    assert polygons_to_boxes([[2, square]]) == ["2 0.300000 0.300000 0.400000 0.400000"]
    print("selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images")
    ap.add_argument("--labels", help="YOLO-seg files to start from (prelabels); never modified")
    ap.add_argument("--out", help="where corrected YOLO-seg files are written")
    ap.add_argument("--classes", help="classes.txt, one name per line")
    ap.add_argument("--match", default="*", help="only label files whose stem matches this glob, e.g. 'plantdoc_*'")
    ap.add_argument("--review", help="optional review.csv: suspicious images first, flags in the status bar")
    ap.add_argument("--export-boxes", metavar="DIR", help="write YOLO detection boxes from the reviewed polygons in --out, then exit")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.export_boxes:
        if not (args.out and args.classes):
            ap.error("--export-boxes needs --out and --classes")
        export_boxes(Path(args.out), Path(args.export_boxes), Path(args.classes))
    else:
        missing = [k for k in ("images", "labels", "out", "classes") if not getattr(args, k)]
        if missing:
            ap.error("missing " + ", ".join(f"--{k}" for k in missing))
        root = tk.Tk()
        Labeler(root, args)
        root.mainloop()
