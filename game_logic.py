# game_logic.py

import cv2
import time
import random
import os

# ── Config ────────────────────────────────────────────────────────
OBSTACLE_SPEED   = 600     # px/s (used for obstacles, freeze, hearts)
SPAWN_INTERVAL   = 1.5     # base seconds between obstacle spawns
FREEZE_INTERVAL  = 8.0     # seconds between freeze-item spawns
MAX_LIVES        = 3

# Sprite scales (fraction of lane width)
CAR_SCALE        = 0.35
OBS_SCALE        = 0.35

# Colors (BGR)
COLOR_LANE       = (50, 50, 50)


class Car:
    def __init__(self, lane_count):
        self.lane_count = lane_count
        self.lane       = lane_count // 2
        self.rect       = (0, 0, 0, 0)
        self.current_x  = None
        self.sprite     = None  # set by Game

    def update(self, dt, boundaries, frame_h):
        left, right = boundaries[self.lane], boundaries[self.lane+1]
        target_x    = (left + right) // 2
        if self.current_x is None:
            self.current_x = target_x
        else:
            alpha = min(1.0, dt * 10.0)
            self.current_x += int((target_x - self.current_x) * alpha)

        lane_w = right - left
        w      = int(lane_w * CAR_SCALE)
        h      = int(w * 1.7)
        x1     = self.current_x - w//2
        y1     = int(frame_h * 0.85) - h//2
        self.rect = (x1, y1, x1 + w, y1 + h)

    def draw(self, overlay):
        x1, y1, x2, y2 = self.rect
        H, W = overlay.shape[:2]
        if self.sprite is None:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,200,0), -1)
            return

        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        w_c, h_c = x2c-x1c, y2c-y1c
        if w_c <= 0 or h_c <= 0:
            return
        sprite = cv2.resize(self.sprite, (w_c, h_c), interpolation=cv2.INTER_AREA)
        alpha_s = sprite[:, :, 3] / 255.0
        for c in range(3):
            overlay[y1c:y2c, x1c:x2c, c] = (
                alpha_s * sprite[:, :, c] +
                (1 - alpha_s) * overlay[y1c:y2c, x1c:x2c, c]
            )


class Obstacle:
    def __init__(self, lane_count):
        self.lane_count = lane_count
        self.lane       = random.randrange(lane_count)
        self.y          = 0.0
        self.rect       = (0, 0, 0, 0)
        self.sprite     = None  # assigned at spawn

    def update(self, dt, boundaries):
        self.y += OBSTACLE_SPEED * dt
        left, right = boundaries[self.lane], boundaries[self.lane+1]
        lane_w      = right - left
        w           = int(lane_w * OBS_SCALE)
        h           = int(w * 1.7)
        cx          = (left + right)//2
        x1          = cx - w//2
        y1          = int(self.y) - h//2
        self.rect   = (x1, y1, x1 + w, y1 + h)

    def draw(self, overlay):
        x1, y1, x2, y2 = self.rect
        H, W = overlay.shape[:2]
        if self.sprite is None:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,0,200), -1)
            return

        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        w_c, h_c = x2c-x1c, y2c-y1c
        if w_c <= 0 or h_c <= 0:
            return
        sprite = cv2.resize(self.sprite, (w_c, h_c), interpolation=cv2.INTER_AREA)
        alpha_s = sprite[:, :, 3] / 255.0
        for c in range(3):
            overlay[y1c:y2c, x1c:x2c, c] = (
                alpha_s * sprite[:, :, c] +
                (1 - alpha_s) * overlay[y1c:y2c, x1c:x2c, c]
            )


class FreezeItem:
    def __init__(self, lane_count):
        self.lane_count = lane_count
        self.lane       = 0
        self.y          = 0.0
        self.rect       = (0, 0, 0, 0)

    def update(self, dt, boundaries):
        self.y += OBSTACLE_SPEED * dt
        left, right = boundaries[self.lane], boundaries[self.lane+1]
        r    = int((right - left) * 0.15)
        cx   = (left + right)//2
        cy   = int(self.y)
        self.rect = (cx-r, cy-r, cx+r, cy+r)

    def draw(self, overlay):
        x1, y1, x2, y2 = self.rect
        cv2.circle(overlay,
                   ((x1+x2)//2, (y1+y2)//2),
                   (x2-x1)//2, COLOR_LANE, -1)


class HeartItem:
    def __init__(self, lane_count):
        self.lane_count = lane_count
        self.lane       = 0
        self.y          = 0.0
        self.rect       = (0, 0, 0, 0)
        self.sprite     = None

    def update(self, dt, boundaries):
        self.y += OBSTACLE_SPEED * dt
        left, right = boundaries[self.lane], boundaries[self.lane+1]
        size        = int((right - left) * 0.2)
        cx, cy      = (left + right)//2, int(self.y)
        self.rect   = (cx-size, cy-size, cx+size, cy+size)

    def draw(self, overlay):
        if self.sprite is None:
            return
        x1, y1, x2, y2 = self.rect
        H, W = overlay.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        w_c, h_c = x2c-x1c, y2c-y1c
        if w_c <= 0 or h_c <= 0:
            return
        sprite = cv2.resize(self.sprite, (w_c, h_c), interpolation=cv2.INTER_AREA)
        alpha_s = sprite[:, :, 3] / 255.0
        for c in range(3):
            overlay[y1c:y2c, x1c:x2c, c] = (
                alpha_s * sprite[:, :, c] +
                (1 - alpha_s) * overlay[y1c:y2c, x1c:x2c, c]
            )


class Game:
    def __init__(self, lane_count):
        self.lane_count      = lane_count
        self.car             = Car(lane_count)
        self.obstacles       = []
        self.freezes         = []
        self.hearts          = []
        self.lives           = MAX_LIVES
        self.score           = 0
        self.spawn_interval  = SPAWN_INTERVAL
        self.freeze_interval = FREEZE_INTERVAL
        t = time.time()
        self.t_last   = t
        self.t_obs    = t
        self.t_freeze = t
        self.boundaries= [0]*(lane_count+1)

        base   = os.path.dirname(__file__)
        imgdir = os.path.join(base, "images")

        self.car.sprite     = cv2.imread(os.path.join(imgdir, "main.png"), cv2.IMREAD_UNCHANGED)
        self.obs_sprites    = [cv2.imread(os.path.join(imgdir, fn), cv2.IMREAD_UNCHANGED)
                               for fn in ("car-truck2.png","car-truck3.png","car-truck4.png","car-truck5.png")]
        self.obs_sprites    = [img for img in self.obs_sprites if img is not None]
        self.heart_sprite   = cv2.imread(os.path.join(imgdir, "heart.png"), cv2.IMREAD_UNCHANGED)

        # load highscore
        self.highscore_file = os.path.join(base, "highscore.txt")
        if os.path.isfile(self.highscore_file):
            try:
                self.high_score = int(open(self.highscore_file).read().strip())
            except (ValueError, IOError):
                self.high_score = 0
        else:
            self.high_score = 0

    def on_blink(self):
        old = self.spawn_interval
        self.spawn_interval = max(0.5, self.spawn_interval * 0.9)

        print(f">>> Blink! {old:.2f}s→{self.spawn_interval:.2f}s")

    def min_spawn(self):
        # ensure enough vertical gap: obstacle height / speed
        lane_w = (self.boundaries[1] - self.boundaries[0])
        h      = int((lane_w * OBS_SCALE) * 1.7)
        return h / OBSTACLE_SPEED * 1.2

    def _compute_boundaries(self, frame_w):
        base   = frame_w // self.lane_count
        rem    = frame_w %  self.lane_count
        widths = [base + (1 if i < rem else 0) for i in range(self.lane_count)]
        b = [0]
        for wi in widths:
            b.append(b[-1] + wi)
        self.boundaries = b

    def _spawn_obstacle(self, now):
        if now - self.t_obs < self.spawn_interval:
            return
        obs = Obstacle(self.lane_count)
        obs.sprite = random.choice(self.obs_sprites)
        self.obstacles.append(obs)
        self.t_obs = now
            # ← removed any random factor here


    def _spawn_freeze(self, now):
        if now - self.t_freeze < self.freeze_interval:
            return
        # pick a lane free in top region
        top_thresh = 0.2  # 20%
        occupied = {o.lane for o in self.obstacles if o.rect[3] < self.boundaries[0] + (self.boundaries[-1] * top_thresh)}
        occupied |= {f.lane for f in self.freezes   if f.rect[3] < self.boundaries[0] + (self.boundaries[-1] * top_thresh)}
        occupied |= {h.lane for h in self.hearts    if h.rect[3] < self.boundaries[0] + (self.boundaries[-1] * top_thresh)}
        free_lanes = list(set(range(self.lane_count)) - occupied)
        lane = random.choice(free_lanes) if free_lanes else random.randrange(self.lane_count)
        fi = FreezeItem(self.lane_count)
        fi.lane = lane
        self.freezes.append(fi)
        self.t_freeze = now

    def _update_obstacles(self, dt, frame_h):
        for obs in self.obstacles[:]:
            obs.update(dt, self.boundaries)
            if obs.rect[1] > frame_h:
                self.obstacles.remove(obs)
                self.score += 1
            elif Game._intersect(obs.rect, self.car.rect):
                self.lives -= 1
                self.obstacles.remove(obs)
                if self.lives < MAX_LIVES:
                    # spawn a collectible heart
                    top_thresh = 0.2
                    occupied = {o.lane for o in self.obstacles if o.rect[3] < frame_h*top_thresh}
                    occupied |= {f.lane for f in self.freezes   if f.rect[3] < frame_h*top_thresh}
                    free_lanes = list(set(range(self.lane_count)) - occupied)
                    lane = random.choice(free_lanes) if free_lanes else random.randrange(self.lane_count)
                    ht = HeartItem(self.lane_count)
                    ht.sprite = self.heart_sprite
                    ht.lane   = lane
                    self.hearts.append(ht)

    def _update_freezes(self, dt, frame_h):
        for fr in self.freezes[:]:
            fr.update(dt, self.boundaries)
            if fr.rect[1] > frame_h:
                self.freezes.remove(fr)
            elif Game._intersect(fr.rect, self.car.rect):
                old = self.spawn_interval
                self.spawn_interval += 0.5
                print(f">>> Freeze! {old:.2f}s→{self.spawn_interval:.2f}s")
                self.freezes.remove(fr)

    def _update_hearts(self, dt, frame_h):
        for ht in self.hearts[:]:
            ht.update(dt, self.boundaries)
            if ht.rect[1] > frame_h:
                self.hearts.remove(ht)
            elif Game._intersect(ht.rect, self.car.rect):
                self.lives = min(MAX_LIVES, self.lives + 1)
                self.hearts.remove(ht)

    def update(self, player_lane, frame_w, frame_h):
        now = time.time()
        dt  = now - self.t_last
        self.t_last = now

        self._compute_boundaries(frame_w)
        self.car.lane = player_lane
        self.car.update(dt, self.boundaries, frame_h)

        self._spawn_obstacle(now)
        self._spawn_freeze(now)

        self._update_obstacles(dt, frame_h)
        self._update_freezes(dt, frame_h)
        self._update_hearts(dt, frame_h)

    def draw(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        alpha   = 0.6

        # draw lanes
        for x in self.boundaries[1:-1]:
            cv2.line(overlay, (x, 0), (x, h), COLOR_LANE, 1)

        # draw items
        for obs in self.obstacles:   obs.draw(overlay)
        for fr  in self.freezes:     fr.draw(overlay)
        for ht  in self.hearts:      ht.draw(overlay)
        self.car.draw(overlay)

        # HUD: lives
        for i in range(self.lives):
            cv2.circle(overlay, (w - 50 - i*40, 50), 15, (0,0,255), -1)

        # HUD: current score
        cv2.putText(overlay, f"SCORE: {self.score}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def save_highscore(self):
        if self.score > self.high_score:
            with open(self.highscore_file, "w") as f:
                f.write(str(self.score))
            self.high_score = self.score

    @staticmethod
    def _intersect(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)
