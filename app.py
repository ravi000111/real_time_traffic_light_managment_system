"""
Traffic Detection + Fixed-Cycle Adaptive Signal Controller
===========================================================

Signal logic
------------
* A CYCLE BUDGET (default 120 s) is divided among all lanes.
* At the START of every cycle the controller samples each lane's
  smoothed vehicle count and computes, for EVERY lane:
      green_i  = clamp( ratio_i * green_budget, MIN_GREEN, MAX_GREEN )
      yellow_i = YELLOW_SEC  (fixed)
      red_i    = total_cycle - green_i - yellow_i
* Phases execute: L0-green → L0-yellow → L1-green → L1-yellow → L2-green → L2-yellow
* After the last lane's yellow the next cycle begins with a fresh sample.
* No lane waits more than (cycle_budget - its own green - its own yellow).

Fast-forward fix
-----------------
Reader thread sleeps wall-clock time equal to 1/fps between reads.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os, sys, threading, time
from collections import deque

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
MODEL_PATH = "/home/ravis/projects/TLM/Final Project/runs/detect/yolov8s_run/weights/best.pt"

VIDEO_PATHS = [
    "/mnt/c/Users/ravis/Videos/VIdeo/ved/Converted_folder/100 futa pillibhit fix 1 morning.mp4",
    "/mnt/c/Users/ravis/Videos/VIdeo/ved/Converted_folder/100 FUTA PILLIBHIT FIX 3 AFTERNOON.mp4",
    "/mnt/c/Users/ravis/Videos/VIdeo/ved/Converted_folder/100 FUTA PILLIBHIT FIX 4 EVENING.mp4",
]

# ─────────────────────────────────────────────
#  DETECTION SETTINGS
# ─────────────────────────────────────────────
VEHICLE_CLASS_NAMES = ['Auto', 'Bike', 'Bus', 'Car', 'Truck']

ORIG_ROI_POLYGONS = [
    [(210, 710), (500,  49), (558,  40), (1265, 659)],
    [(264, 717), (549,  87), (634,  87), (1128, 693)],
    [(484, 712), (660,  87), (739,  81), (1103, 698)],
]
REF_W, REF_H   = 1280, 720
PROCESS_WIDTH  = 640
PROCESS_HEIGHT = 360
DISPLAY_WIDTH  = 640
DISPLAY_HEIGHT = 480
CONF_THRESHOLD = 0.4
IMG_SIZE       = 640
BUFFER_SIZE    = 8

# ─────────────────────────────────────────────
#  SIGNAL SETTINGS  ← tune these
# ─────────────────────────────────────────────
CYCLE_BUDGET_SEC = 120      # total cycle length in seconds
MIN_GREEN_SEC    = 10       # every lane gets at least this
MAX_GREEN_SEC    = 60       # no lane gets more than this
YELLOW_SEC       = 3        # fixed yellow between green→red
DENSITY_WINDOW   = 20       # frames averaged for smoothing

# Signal state constants
GREEN  = "GREEN"
YELLOW = "YELLOW"
RED    = "RED"

# BGR colours for overlay
CLR = {
    GREEN:  (0,  200,   0),
    YELLOW: (0,  220, 220),
    RED:    (0,    0, 200),
    "WHITE":(255, 255, 255),
    "BLACK":(  0,   0,   0),
}

# ─────────────────────────────────────────────
#  VALIDATE PATHS
# ─────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    sys.exit(f"❌  Model not found:\n    {MODEL_PATH}")
for vp in VIDEO_PATHS:
    if not os.path.exists(vp):
        print(f"⚠️  Video not found (will show blank): {vp}")

USE_GUI = bool(os.environ.get("DISPLAY", ""))
if not USE_GUI:
    print("⚠️  No DISPLAY – headless mode. Saving to ./output_frames/")
    os.makedirs("output_frames", exist_ok=True)

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"ℹ️  Device: {device}")
model = YOLO(MODEL_PATH)
model.to(device)
model_lock = threading.Lock()

class_names           = model.names
vehicle_class_indices = {i for i, n in class_names.items() if n in VEHICLE_CLASS_NAMES}
print(f"ℹ️  Vehicle class indices: {vehicle_class_indices}")

# ─────────────────────────────────────────────
#  ROI POLYGONS  (pre-scaled once)
# ─────────────────────────────────────────────
num_lanes = len(VIDEO_PATHS)

def scale_polygon(poly, ow, oh, tw, th):
    return [(int(x * tw / ow), int(y * th / oh)) for x, y in poly]

roi_polys_proc = [
    np.array(scale_polygon(ORIG_ROI_POLYGONS[i], REF_W, REF_H,
                           PROCESS_WIDTH, PROCESS_HEIGHT), dtype=np.int32)
    for i in range(num_lanes)
]

# ─────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────
latest_display  = [np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8)
                   for _ in range(num_lanes)]
latest_counts   = [0] * num_lanes
lane_locks      = [threading.Lock() for _ in range(num_lanes)]
lane_done       = [False] * num_lanes
stop_event      = threading.Event()

count_history   = [deque(maxlen=DENSITY_WINDOW) for _ in range(num_lanes)]
count_hist_lock = threading.Lock()

# ─────────────────────────────────────────────
#  FIXED-CYCLE SIGNAL CONTROLLER
# ─────────────────────────────────────────────
class SignalController:
    """
    At the start of each cycle:
      1. Sample smoothed density for every lane.
      2. Compute green_i proportionally within [MIN_GREEN_SEC, MAX_GREEN_SEC].
         The green_budget = CYCLE_BUDGET_SEC - num_lanes * YELLOW_SEC.
      3. red_i = cycle_budget - green_i - yellow_i  (pre-computed & displayed).
      4. Execute phases in round-robin order; then start the next cycle.

    Public read:
      get_state() → (signals list, timers list, plan dict)
        signals[i] : GREEN / YELLOW / RED
        timers[i]  : seconds remaining in the *current phase for that lane*
        plan[i]    : {'green': g, 'yellow': y, 'red': r}  for current cycle
    """

    def __init__(self, n: int):
        self.n        = n
        self._signals = [RED] * n
        self._timers  = [0.0] * n
        self._plan    = [{'green': 0.0, 'yellow': float(YELLOW_SEC), 'red': 0.0}
                         for _ in range(n)]
        self._lock    = threading.Lock()
        t = threading.Thread(target=self._run, daemon=True, name="signal-ctrl")
        t.start()

    def get_state(self):
        with self._lock:
            return (list(self._signals),
                    list(self._timers),
                    [dict(p) for p in self._plan])

    # ── density helpers ───────────────────────────────────────────────
    def _smoothed_counts(self):
        with count_hist_lock:
            return [
                (sum(count_history[i]) / len(count_history[i]))
                if count_history[i] else 0.0
                for i in range(self.n)
            ]

    def _compute_plan(self, counts):
        total = sum(counts)
        green_budget = CYCLE_BUDGET_SEC - self.n * YELLOW_SEC   # seconds available for green

        if total == 0:
            ratios = [1.0 / self.n] * self.n
        else:
            ratios = [c / total for c in counts]

        # Scale so no lane exceeds MAX_GREEN_SEC, but preserve ratios
        max_r = max(ratios)
        scale = min(MAX_GREEN_SEC / (max_r * green_budget), 1.0) if max_r > 0 else 1.0

        plan = []
        for r in ratios:
            g = max(MIN_GREEN_SEC, min(MAX_GREEN_SEC, r * green_budget * scale))
            y = float(YELLOW_SEC)
            # Red = everything else in the cycle that isn't THIS lane's green+yellow
            # = sum of all other lanes' (green + yellow)
            # We compute it after finalising all g values; placeholder for now.
            plan.append({'green': g, 'yellow': y, 'red': 0.0})

        # Fill in red times
        for i in range(self.n):
            others_time = sum(
                plan[j]['green'] + plan[j]['yellow']
                for j in range(self.n) if j != i
            )
            plan[i]['red'] = others_time

        return plan

    # ── controller loop ───────────────────────────────────────────────
    def _run(self):
        cycle_num = 0

        while not stop_event.is_set():
            cycle_num += 1
            counts = self._smoothed_counts()
            plan   = self._compute_plan(counts)

            # Publish the plan
            with self._lock:
                self._plan = plan

            print(
                f"\n🚦 Cycle {cycle_num} starts  (budget={CYCLE_BUDGET_SEC}s)"
            )
            for i, (p, c) in enumerate(zip(plan, counts)):
                print(f"   Lane {i}: avg={c:.1f}v  "
                      f"green={p['green']:.1f}s  "
                      f"yellow={p['yellow']:.1f}s  "
                      f"red={p['red']:.1f}s")

            # Execute round-robin phases
            for active in range(self.n):
                if stop_event.is_set():
                    break

                g = plan[active]['green']
                y = plan[active]['yellow']

                # ── GREEN phase ───────────────────────────────────────
                end = time.time() + g
                while time.time() < end and not stop_event.is_set():
                    now = time.time()
                    rem_active = max(0.0, end - now)
                    with self._lock:
                        for i in range(self.n):
                            if i == active:
                                self._signals[i] = GREEN
                                self._timers[i]  = rem_active
                            else:
                                self._signals[i] = RED
                                # Waiting time = time until THIS lane's green starts:
                                # remaining green+yellow of active lane
                                # + green+yellow of all lanes between active and i
                                wait = rem_active + plan[active]['yellow']
                                for j in range(self.n):
                                    lane_j = (active + 1 + j) % self.n
                                    if lane_j == i:
                                        break
                                    wait += plan[lane_j]['green'] + plan[lane_j]['yellow']
                                self._timers[i] = max(0.0, wait)
                    time.sleep(0.05)

                # ── YELLOW phase ──────────────────────────────────────
                end = time.time() + y
                while time.time() < end and not stop_event.is_set():
                    now = time.time()
                    rem_active = max(0.0, end - now)
                    with self._lock:
                        for i in range(self.n):
                            if i == active:
                                self._signals[i] = YELLOW
                                self._timers[i]  = rem_active
                            else:
                                self._signals[i] = RED
                                wait = rem_active
                                for j in range(self.n):
                                    lane_j = (active + 1 + j) % self.n
                                    if lane_j == i:
                                        break
                                    wait += plan[lane_j]['green'] + plan[lane_j]['yellow']
                                self._timers[i] = max(0.0, wait)
                    time.sleep(0.05)

            # Brief all-red gap between cycles
            with self._lock:
                for i in range(self.n):
                    self._signals[i] = RED
                    self._timers[i]  = 0.0
            time.sleep(0.5)


# ─────────────────────────────────────────────
#  OVERLAY HELPERS
# ─────────────────────────────────────────────
def draw_signal_box(frame, signal: str, timer: float):
    """Top-right corner: coloured signal box with countdown for ALL states."""
    sig_color = CLR.get(signal, CLR["WHITE"])
    bx, by, bw, bh = DISPLAY_WIDTH - 140, 8, 132, 80

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), CLR["BLACK"], -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), sig_color, 2)

    cv2.putText(frame, signal,
                (bx + 10, by + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, sig_color, 2)

    if timer > 0:
        label = f"wait {timer:.0f}s" if signal == RED else f"{timer:.1f}s"
        cv2.putText(frame, label,
                    (bx + 8, by + 63),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, CLR["WHITE"], 2)


def draw_plan_bar(frame, plan: dict, signal: str):
    """
    Bottom strip: horizontal bar showing green / yellow / red proportions
    for THIS lane in the current cycle.
    """
    g, y, r = plan['green'], plan['yellow'], plan['red']
    total = g + y + r
    if total == 0:
        return

    bar_x, bar_y = 5, DISPLAY_HEIGHT - 22
    bar_w        = DISPLAY_WIDTH - 10
    bar_h        = 14

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (30, 30, 30), -1)

    gw = int(bar_w * g / total)
    yw = int(bar_w * y / total)
    rw = bar_w - gw - yw

    cv2.rectangle(frame, (bar_x,          bar_y),
                         (bar_x + gw,      bar_y + bar_h), (0, 180, 0),   -1)
    cv2.rectangle(frame, (bar_x + gw,      bar_y),
                         (bar_x + gw + yw, bar_y + bar_h), (0, 200, 200), -1)
    cv2.rectangle(frame, (bar_x + gw + yw, bar_y),
                         (bar_x + bar_w,   bar_y + bar_h), (0, 0, 180),   -1)

    # Labels inside bar
    cv2.putText(frame, f"G:{g:.0f}s",
                (bar_x + 3, bar_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.putText(frame, f"Y:{y:.0f}s",
                (bar_x + gw + 3, bar_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
    cv2.putText(frame, f"R:{r:.0f}s",
                (bar_x + gw + yw + 3, bar_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)


def draw_lane_header(frame, lane_idx: int, count: int, signal: str, timer: float, plan: dict):
    """Composite annotation: semi-transparent header + signal box + plan bar."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (DISPLAY_WIDTH, 48), CLR["BLACK"], -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Lane {lane_idx}  |  {count} vehicles",
                (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.72, CLR["WHITE"], 2)

    draw_signal_box(frame, signal, timer)
    draw_plan_bar(frame, plan, signal)


# ─────────────────────────────────────────────
#  LANE WORKER THREAD
# ─────────────────────────────────────────────
def lane_worker(lane_idx: int):
    path   = VIDEO_PATHS[lane_idx]
    roi_np = roi_polys_proc[lane_idx]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"❌  Lane {lane_idx}: cannot open video")
        lane_done[lane_idx] = True
        return

    native_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_delay = 1.0 / native_fps          # seconds per frame
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅  Lane {lane_idx}: {w}×{h}  {native_fps:.1f} fps  "
          f"frame_delay={frame_delay*1000:.1f} ms")

    frame_buf: deque = deque(maxlen=BUFFER_SIZE)

    # ── Reader sub-thread: throttled to native FPS ────────────────────
    def reader():
        last_read = time.perf_counter()
        while not stop_event.is_set():
            # Sleep exactly until the next frame is due
            due = last_read + frame_delay
            gap = due - time.perf_counter()
            if gap > 0:
                time.sleep(gap)
            ret, frm = cap.read()
            last_read = time.perf_counter()
            if not ret:
                break
            frame_buf.append(frm)
        frame_buf.append(None)   # sentinel

    reader_t = threading.Thread(target=reader, daemon=True)
    reader_t.start()

    # ── Inference loop ────────────────────────────────────────────────
    while not stop_event.is_set():
        frame = None
        while frame_buf:
            frame = frame_buf.popleft()

        if frame is None:
            time.sleep(0.005)
            continue
        if not isinstance(frame, np.ndarray):
            break   # sentinel

        proc = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        rgb  = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

        with model_lock:
            results = model(rgb, device=device, imgsz=IMG_SIZE,
                            conf=CONF_THRESHOLD, verbose=False)

        count = 0
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in vehicle_class_indices:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                inside = cv2.pointPolygonTest(roi_np, (cx, cy), False) >= 0
                if inside:
                    count += 1
                cv2.rectangle(proc,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0) if inside else (64, 128, 0), 2)
                cv2.putText(proc, class_names[cls_id],
                            (int(x1), max(int(y1) - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 0), 1)

        # Update density history
        with count_hist_lock:
            count_history[lane_idx].append(count)
        smoothed = int(round(
            sum(count_history[lane_idx]) / len(count_history[lane_idx])
        ))

        cv2.polylines(proc, [roi_np], True, (0, 255, 255), 2)
        display = cv2.resize(proc, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        with lane_locks[lane_idx]:
            latest_display[lane_idx] = display
            latest_counts[lane_idx]  = smoothed

    cap.release()
    lane_done[lane_idx] = True
    print(f"ℹ️  Lane {lane_idx} finished.")


# ─────────────────────────────────────────────
#  START LANE THREADS + CONTROLLER
# ─────────────────────────────────────────────
workers = []
for i in range(num_lanes):
    t = threading.Thread(target=lane_worker, args=(i,),
                         daemon=True, name=f"lane-{i}")
    t.start()
    workers.append(t)

controller = SignalController(num_lanes)

# ─────────────────────────────────────────────
#  DISPLAY LOOP
# ─────────────────────────────────────────────
saved_idx = 0
print("ℹ️  Press  Q  to quit.\n")

while not stop_event.is_set():
    signals, timers, plan = controller.get_state()

    frames = []
    for i in range(num_lanes):
        with lane_locks[i]:
            frame = latest_display[i].copy()
            count = latest_counts[i]
        draw_lane_header(frame, i, count, signals[i], timers[i], plan[i])
        frames.append(frame)

    combined = np.hstack(frames)

    # ── Status bar (bottom strip across all lanes) ────────────────────
    bar_h  = 28
    bar    = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
    parts  = []
    sym    = {GREEN: "G", YELLOW: "Y", RED: "R"}
    for i in range(num_lanes):
        p = plan[i]
        parts.append(
            f"L{i}[{latest_counts[i]}v] "
            f"{sym.get(signals[i], '?')}:{timers[i]:.0f}s "
            f"G={p['green']:.0f} Y={p['yellow']:.0f} R={p['red']:.0f}"
        )
    cv2.putText(bar, "  |  ".join(parts),
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    combined = np.vstack([combined, bar])

    if USE_GUI:
        cv2.imshow("Traffic System", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit.")
            stop_event.set()
            break
    else:
        saved_idx += 1
        if saved_idx % 30 == 0:
            out = f"output_frames/frame_{saved_idx:06d}.jpg"
            cv2.imwrite(out, combined)
            print(f"[{saved_idx}] {out}  signals={signals}")

    if all(lane_done):
        print("All lanes finished.")
        stop_event.set()
        break

    time.sleep(0.01)

# ─────────────────────────────────────────────
#  CLEANUP
# ─────────────────────────────────────────────
stop_event.set()
for t in workers:
    t.join(timeout=5)
if USE_GUI:
    cv2.destroyAllWindows()

print(f"\nFinal counts : {latest_counts}")
