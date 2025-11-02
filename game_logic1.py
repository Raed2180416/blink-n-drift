import os
import time
import random
import cv2
import numpy as np

# --- Game Difficulty ---
INITIAL_OBSTACLE_SPEED = 350
SPEED_INCREMENT = 20        # Speed increase per blink
SPAWN_INTERVAL = 2.0        # Initial time between obstacle spawns

# --- Collectibles & Player Stats ---
COIN_INTERVAL = 0.8         # Reduced from 3.0 for more coins
HEART_INTERVAL = 12.0
MAX_LIVES = 3

# --- Object Visual Scaling ---
CAR_SCALE = 0.28
OBS_SCALE = 0.28
COIN_SCALE = 0.15
HEART_SCALE = 0.23

# --- Layout & Spawning ---
COLOR_LANE = (50, 50, 50)
MIN_VERTICAL_GAP = 400      # Prevents items from overlapping
SPAWN_Y = -50
OBSTACLE_SPAWN_LOOKAHEAD_SECONDS = 0.45 # Balanced value: not too safe, not too chaotic

# --- Collision Forgiveness ---
# Shrinks hitboxes. 0.2 means hitbox is 20% smaller than the sprite.
CAR_V_FORGIVENESS = 0.35
OBS_V_FORGIVENESS = 0.35
CAR_H_FORGIVENESS = 0.25
OBS_H_FORGIVENESS = 0.25

# --- Game Modes ---
PATTERN_DURATION = 20 # Each pattern will last 20 seconds
# REMOVED: BONUS_DURATION = 10
PATTERN_COOLDOWN_S = 3.0 # 3-second relaxation buffer between patterns


OBSTACLE_PATTERNS = {
    "zigzag": [
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ],
    "wall_gap": [
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 1],
    ],
    "alternating": [
        [1, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    "funnel": [
        [1, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    "spiral": [
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
}

def _calculate_blit_coords(item_x_in_overlay, item_y_in_overlay, item_w, item_h, overlay_w, overlay_h):
    dst_x1 = max(0, item_x_in_overlay)
    dst_y1 = max(0, item_y_in_overlay)
    dst_x2 = min(overlay_w, item_x_in_overlay + item_w)
    dst_y2 = min(overlay_h, item_y_in_overlay + item_h)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return None

    src_x1 = dst_x1 - item_x_in_overlay
    src_y1 = dst_y1 - item_y_in_overlay
    src_x1 = max(0, src_x1)
    src_y1 = max(0, src_y1)
    
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    
    return (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2)

def draw_sprite(sprite, rect, overlay):
    x1, y1, x2, _ = map(int, rect)
    H, W = overlay.shape[:2]
    
    orig_h, orig_w = sprite.shape[:2]
    if orig_h == 0: return
    aspect_ratio = orig_w / float(orig_h)
    
    target_w = x2 - x1
    if target_w <= 0: return

    if aspect_ratio == 0:
        target_h = target_w 
    else:
        target_h = int(target_w / aspect_ratio)
    
    if target_h <= 0: return

    y1_new = y1
    
    try:
        resized = cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_AREA)
    except cv2.error:
        return
    
    blit_details = _calculate_blit_coords(x1, y1_new, target_w, target_h, W, H)
    if blit_details is None:
        return

    (src_x1_sprite, src_y1_sprite, src_x2_sprite, src_y2_sprite), \
    (dst_x1_overlay, dst_y1_overlay, dst_x2_overlay, dst_y2_overlay) = blit_details
    
    src_x1_sprite = max(0, src_x1_sprite)
    src_y1_sprite = max(0, src_y1_sprite)
    src_x2_sprite = min(resized.shape[1], src_x2_sprite)
    src_y2_sprite = min(resized.shape[0], src_y2_sprite)

    if src_x2_sprite <= src_x1_sprite or src_y2_sprite <= src_y1_sprite:
         return

    visible_portion = resized[src_y1_sprite:src_y2_sprite, src_x1_sprite:src_x2_sprite]
    
    if visible_portion.size == 0:
        return

    if visible_portion.shape[2] == 4: 
        # --- OPTIMIZED: High-speed alpha blending with NumPy ---
        alpha_mask = visible_portion[:, :, 3, np.newaxis] / 255.0
        target_region = overlay[dst_y1_overlay:dst_y2_overlay, dst_x1_overlay:dst_x2_overlay]
        
        blended = (visible_portion[:, :, :3] * alpha_mask + target_region * (1 - alpha_mask)).astype(overlay.dtype)
        overlay[dst_y1_overlay:dst_y2_overlay, dst_x1_overlay:dst_x2_overlay] = blended
    else: 
        overlay[dst_y1_overlay:dst_y2_overlay, dst_x1_overlay:dst_x2_overlay] = visible_portion

class Car:
    def __init__(self, lane_count):
        self.lane_count = lane_count
        self.lane = lane_count // 2
        self.rect = (0, 0, 0, 0)
        self.current_x = None
        self.sprite = None

    def update(self, dt, boundaries, frame_h):
        left, right = boundaries[self.lane], boundaries[self.lane+1]
        target_x = (left + right) // 2
        if self.current_x is None:
            self.current_x = target_x
        else:
            alpha = min(1.0, dt * 10.0)
            self.current_x += int((target_x - self.current_x) * alpha)
        lane_w = right - left
        w = int(lane_w * CAR_SCALE)
        h = int(w * 1.7)
        x1 = self.current_x - w//2
        y1 = int(frame_h * 0.85) - h//2
        self.rect = (x1, y1, x1 + w, y1 + h)

    def draw(self, overlay):
        if self.sprite is None:
            return
        draw_sprite(self.sprite, self.rect, overlay)

class Obstacle:
    def __init__(self, lane_count, sprite):
        self.lane = random.randint(0, lane_count-1)
        self.y = SPAWN_Y
        self.sprite = sprite
        self.rect = (0, 0, 0, 0)

    def update(self, dt, boundaries, speed): # ADDED speed parameter
        self.y += speed * dt # CHANGED to use passed-in speed
        left, right = boundaries[self.lane], boundaries[self.lane+1]
        lane_w = right - left
        
        w = int(lane_w * OBS_SCALE)
        
        if self.sprite is not None and self.sprite.shape[0] > 0 and self.sprite.shape[1] > 0:
            sprite_orig_h, sprite_orig_w = self.sprite.shape[:2]
            if sprite_orig_h == 0:
                 h = int(w * 1.7)
            else:
                sprite_aspect_ratio = float(sprite_orig_w) / float(sprite_orig_h)
                if sprite_aspect_ratio > 0:
                    h = int(w / sprite_aspect_ratio)
                else:
                    h = int(w * 1.7)
        else:
            h = int(w * 1.7)
            
        cx = (left + right)//2
        x1 = cx - w//2
        y1 = int(self.y) - h//2
        self.rect = (x1, y1, x1 + w, y1 + h)

    def draw(self, overlay):
        if self.sprite is None:
            return
        draw_sprite(self.sprite, self.rect, overlay)

class CoinItem:
    def __init__(self, lane_count, sprite):
        self.lane_count = lane_count
        self.lane = random.randrange(lane_count)
        self.y = SPAWN_Y
        self.rect = (0, 0, 0, 0)
        self.sprite = sprite

    def update(self, dt, boundaries, speed): # ADDED speed parameter
        self.y += speed * dt # CHANGED to use passed-in speed
        left, right = boundaries[self.lane], boundaries[self.lane + 1]
        r = int((right - left) * COIN_SCALE)
        cx = (left + right) // 2
        cy = int(self.y)
        self.rect = (cx - r, cy - r, cx + r, cy + r)

    def draw(self, overlay):
        if self.sprite is None:
            return
        draw_sprite(self.sprite, self.rect, overlay)

# FreezeItem class removed

class HeartItem:
    def __init__(self, lane_count, sprite):
        self.lane_count = lane_count
        self.lane = random.randrange(lane_count)
        self.y = SPAWN_Y
        self.rect = (0, 0, 0, 0)
        self.sprite = sprite

    def update(self, dt, boundaries, speed): # ADDED speed parameter
        self.y += speed * dt # CHANGED to use passed-in speed
        left, right = boundaries[self.lane], boundaries[self.lane + 1]
        size = int((right - left) * HEART_SCALE)
        radius = size // 2
        cx, cy = (left + right) // 2, int(self.y)
        self.rect = (cx - radius, cy - radius, cx + radius, cy + radius)

    def draw(self, overlay):
        if self.sprite is None:
            return
        draw_sprite(self.sprite, self.rect, overlay)

class ExplosionEffect:
    def __init__(self, initial_rect, sprite):
        self.rect = tuple(initial_rect)
        self.sprite = sprite

    def update(self, dt, speed): # ADDED speed parameter
        dy = speed * dt # CHANGED to use passed-in speed
        self.rect = (self.rect[0], self.rect[1] + dy, self.rect[2], self.rect[3] + dy)

    def draw(self, overlay):
        if self.sprite is None:
            return
        draw_sprite(self.sprite, self.rect, overlay)

class Game:
    def __init__(self, lane_count):
        self.lane_count = lane_count
        self.car = Car(lane_count)
        self.obstacles = []
        self.coins = []

        self.hearts = []
        self.explosion_effects = []
        self.lives = MAX_LIVES
        self.score = 0
        self.coin_count = 0
        self.spawn_interval = SPAWN_INTERVAL

        self.coin_interval = COIN_INTERVAL
        self.heart_interval = HEART_INTERVAL
        t = time.time()
        self.t_last_obs_spawn = t
        self.t_last_coin_spawn = t

        self.t_last_heart_spawn = t
        self.boundaries = [0]*(lane_count+1)
        self.game_over = False

        self.debug_mode = True # SET TO TRUE TO SEE HITBOXES

        # --- NEW: Pattern-cycling engine state ---
        self.pattern_list = list(OBSTACLE_PATTERNS.keys())
        self.pattern_queue = []
        self.pattern_name = None
        self.pattern_step = 0
        self.pattern_spawn_timer = 0
        self.pattern_active = False
        self.pattern_end_time = 0
        self.pattern_cooldown_active = True # Start in cooldown
        self.pattern_cooldown_end_time = time.time() + 3.0 # Initial 3s buffer
        
        # REMOVED: All old pattern/bonus attributes
        # self.pattern_start_score = 0
        # self.bonus_active = False
        # self.bonus_start_time = 0
        # self.original_spawn_interval = self.spawn_interval

        base = os.path.dirname(__file__)
        imgdir = os.path.join(base, "images")

        self.car.sprite = cv2.imread(os.path.join(imgdir, "main.png"), cv2.IMREAD_UNCHANGED)
        
        # --- NEW: Separate car and truck sprites for rarity ---
        all_obs_sprites = [cv2.imread(os.path.join(imgdir, fn), cv2.IMREAD_UNCHANGED)
                           for fn in ("car-truck2.png","car-truck3.png","car-truck4.png","car-truck5.png")]
        all_obs_sprites = [img for img in all_obs_sprites if img is not None]
        
        self.car_sprites = all_obs_sprites[:2]  # First 2 are cars
        self.truck_sprites = all_obs_sprites[2:] # Last 2 are trucks
        self.last_spawn_was_truck = False
        
        self.heart_sprite = cv2.imread(os.path.join(imgdir, "heart.png"), cv2.IMREAD_UNCHANGED)
        self.coin_sprite = cv2.imread(os.path.join(imgdir, "coin.png"), cv2.IMREAD_UNCHANGED)

        self.explosion_sprite = cv2.imread(os.path.join(imgdir, "explosion.png"), cv2.IMREAD_UNCHANGED)
        if self.explosion_sprite is None:
            print("Warning: explosion.png not found in images folder!")


        self.highscore_file = os.path.join(base, "highscore.txt")
        if os.path.isfile(self.highscore_file):
            try:
                with open(self.highscore_file, 'r') as f:
                    self.high_score = int(f.read().strip())
            except (ValueError, IOError):
                self.high_score = 0
        else:
            self.high_score = 0

        self.road_gray = (80, 80, 80)
        self.boundary_th = 6
        self.dash_len = 60
        self.dash_gap = 90
        self.dash_w = 4
        self.dash_col = (255,255,255)
        self.road_alpha = 0.7
        self.road_offset = 0
        self.scroll_speed = 4.0

        self.obstacle_speed = INITIAL_OBSTACLE_SPEED
        
        self._load_next_pattern() # Prepare the first pattern

    def on_blink(self):
        old_interval = self.spawn_interval
        old_speed = self.obstacle_speed
        
        self.spawn_interval = max(0.5, self.spawn_interval * 0.9)
        self.obstacle_speed += SPEED_INCREMENT
        
        print(f">>> Blink! Interval: {old_interval:.2f}s→{self.spawn_interval:.2f}s, Speed: {old_speed}→{self.obstacle_speed}")

    # --- REMOVED: All old spawning logic ---
    # The methods _should_start_pattern, _start_pattern_mode, _start_bonus_stretch,
    # _spawn_bonus_collectibles, _update_bonus_stretch, _spawn_obstacle, and all of
    # its helper functions have been removed.

    # --- NEW: Pattern-cycling engine ---
    def _load_next_pattern(self):
        """Loads the next pattern from the queue and sets its duration."""
        if not self.pattern_queue:
            self.pattern_queue = self.pattern_list[:]
            random.shuffle(self.pattern_queue)
        
        self.pattern_name = self.pattern_queue.pop(0)
        self.pattern_step = 0
        self.pattern_active = True
        self.pattern_cooldown_active = False
        self.pattern_end_time = time.time() + PATTERN_DURATION
        print(f">>> Starting pattern '{self.pattern_name}' for {PATTERN_DURATION}s")

    def _update_pattern_mode(self, now):
        """The primary obstacle spawner, driven by patterns, duration, and cooldowns."""
        if self.pattern_cooldown_active:
            if now > self.pattern_cooldown_end_time:
                self._load_next_pattern()
            return

        if self.pattern_active and now > self.pattern_end_time:
            print(f">>> Pattern '{self.pattern_name}' finished. Starting {PATTERN_COOLDOWN_S}s cooldown.")
            self.pattern_active = False
            self.pattern_cooldown_active = True
            self.pattern_cooldown_end_time = now + PATTERN_COOLDOWN_S
            return

        if not self.pattern_active:
            return
            
        pattern_data = OBSTACLE_PATTERNS[self.pattern_name]
        if self.pattern_step >= len(pattern_data):
            self.pattern_step = 0

        if now - self.pattern_spawn_timer > 0.8:
            current_row = pattern_data[self.pattern_step]
            
            for lane_idx, should_spawn in enumerate(current_row):
                if should_spawn and lane_idx < self.lane_count and self.car_sprites:
                    sprite = random.choice(self.car_sprites)
                    new_obstacle = Obstacle(self.lane_count, sprite)
                    new_obstacle.lane = lane_idx
                    new_obstacle.y = SPAWN_Y
                    self.obstacles.append(new_obstacle)
            
            self.pattern_step += 1
            self.pattern_spawn_timer = now
    
    def _spawn_coin(self, now):
        if self.coin_sprite is None or now - self.t_last_coin_spawn < self.coin_interval:
            return
        lane = self._get_safe_lane(SPAWN_Y)
        if lane is not None:
            coin = CoinItem(self.lane_count, self.coin_sprite)
            coin.lane = lane
            self.coins.append(coin)
            self.t_last_coin_spawn = now

    # _spawn_freeze method removed

    def _spawn_heart(self, now):
        # Only attempt to spawn a heart if lives are not full.
        if self.lives >= MAX_LIVES:
            return
            
        if self.heart_sprite is None or now - self.t_last_heart_spawn < self.heart_interval:
            return
        lane = self._get_safe_lane(SPAWN_Y)
        if lane is not None:
            heart = HeartItem(self.lane_count, self.heart_sprite)
            heart.lane = lane
            self.hearts.append(heart)
            self.t_last_heart_spawn = now

    def _get_safe_lane(self, y_pos):
        safe = []
        for lane in range(self.lane_count):
            if self._is_lane_safe(y_pos, lane):
                safe.append(lane)
        return random.choice(safe) if safe else None

    def update(self, lane, fw, fh):
        if self.game_over:
            return

        self.car.lane = lane
        dt = 1 / 30

        self._initialize_boundaries(fw)
        self._spawn_all_items()
        self.car.update(dt, self.boundaries, fh)

        # REMOVED: self._update_obstacles(fh) was redundant and caused a speed bug.
        self._update_road_animation()
        self._process_game_elements(dt, fh)

    def _update_road_animation(self):
        if hasattr(self, 'dash_len') and hasattr(self, 'dash_gap'):
            cycle = self.dash_len + self.dash_gap
            dt = 1/30
            self.road_offset = (self.road_offset + self.obstacle_speed * dt) % cycle
    
    def _process_explosion_effects(self, dt, fh):
        for i in range(len(self.explosion_effects) - 1, -1, -1):
            effect = self.explosion_effects[i]
            effect.update(dt, self.obstacle_speed) # PASS speed
            if effect.rect[1] > fh:
                self.explosion_effects.pop(i)

    def _process_game_elements(self, dt, fh):
        self._process_obstacles(dt, fh)
        self._process_coins(dt, fh)
        self._process_hearts(dt, fh)
        self._process_explosion_effects(dt, fh)
        self._check_game_over()

    def _initialize_boundaries(self, fw):
        if fw is not None and (not self.boundaries or self.boundaries[1] == 0):
            lane_w = fw // self.lane_count
            self.boundaries = [i * lane_w for i in range(self.lane_count + 1)]
            if self.lane_count > 0 and len(self.boundaries) > self.lane_count:
                self.boundaries[self.lane_count] = fw

    def _spawn_all_items(self):
        now = time.time()
        if self.boundaries and self.boundaries[1] > 0:
            # Obstacles are now handled exclusively by the pattern engine
            self._update_pattern_mode(now)
            
            # FIX: Only spawn collectibles during the relaxation buffer
            if self.pattern_cooldown_active:
                self._spawn_coin(now)
                self._spawn_heart(now)

    def _process_obstacles(self, dt, fh):
        for obs in self.obstacles[:]:
            obs.update(dt, self.boundaries, self.obstacle_speed) # PASS speed
            if obs.rect[1] > fh:
                self._handle_obstacle_scored(obs)
            elif Game._intersect(self.car.rect, obs.rect, forgive_tail=True):
                self._handle_obstacle_collision(obs)

    def _handle_obstacle_scored(self, obs):
        self.obstacles.remove(obs)
        self.score += 1
        if self.score > self.high_score:
            self.high_score = self.score
            self._save_high_score()

    def _save_high_score(self):
        try:
            with open(self.highscore_file, 'w') as f:
                f.write(str(self.high_score))
        except IOError:
            pass

    def _handle_obstacle_collision(self, obs):
        self.lives -= 1
        
        if self.explosion_sprite is not None:
            explosion = ExplosionEffect(tuple(obs.rect), self.explosion_sprite) 
            self.explosion_effects.append(explosion)

        self.obstacles.remove(obs)

    def _process_coins(self, dt, fh):
        for coin in self.coins[:]:
            coin.update(dt, self.boundaries, self.obstacle_speed) # PASS speed
            if coin.rect[1] > fh:
                self.coins.remove(coin)
            elif Game._intersect(coin.rect, self.car.rect, forgive_tail=False):
                self.coin_count += 1
                self.coins.remove(coin)

    # _process_freezes method removed

    def _process_hearts(self, dt, fh):
        for ht in self.hearts[:]:
            ht.update(dt, self.boundaries, self.obstacle_speed) # PASS speed
            if ht.rect[1] > fh:
                self.hearts.remove(ht)
            elif Game._intersect(ht.rect, self.car.rect, forgive_tail=False):
                if self.lives < MAX_LIVES:
                    self.lives += 1
                self.hearts.remove(ht)

    def _check_game_over(self):
        if self.lives <= 0 and not self.game_over:
            self.game_over = True

    def _draw_debug_hitboxes(self, frame):
        """Draws visual and SYMMETRICAL core hitboxes for debugging."""
        # --- Car Hitboxes ---
        ax1, ay1, ax2, ay2 = map(int, self.car.rect)
        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 255), 1) # Yellow (Visual)

        car_w, car_h = ax2 - ax1, ay2 - ay1
        car_h_shrink = (car_w * CAR_H_FORGIVENESS) / 2
        car_v_shrink = (car_h * CAR_V_FORGIVENESS) / 2
        core_ax1 = int(ax1 + car_h_shrink)
        core_ay1 = int(ay1 + car_v_shrink)
        core_ax2 = int(ax2 - car_h_shrink)
        core_ay2 = int(ay2 - car_v_shrink)
        cv2.rectangle(frame, (core_ax1, core_ay1), (core_ax2, core_ay2), (0, 255, 0), 2) # Green (Core)

        # --- Obstacle Hitboxes ---
        for obs in self.obstacles:
            bx1, by1, bx2, by2 = map(int, obs.rect)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 1) # Red (Visual)

            obs_w, obs_h = bx2 - bx1, by2 - by1
            obs_h_shrink = (obs_w * OBS_H_FORGIVENESS) / 2
            obs_v_shrink = (obs_h * OBS_V_FORGIVENESS) / 2
            core_bx1 = int(bx1 + obs_h_shrink)
            core_by1 = int(by1 + obs_v_shrink)
            core_bx2 = int(bx2 - obs_h_shrink)
            core_by2 = int(by2 - obs_v_shrink)
            cv2.rectangle(frame, (core_bx1, core_by1), (core_bx2, core_by2), (0, 255, 0), 2) # Green (Core)

    def draw(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        cv2.rectangle(overlay, (0,0), (w,h), self.road_gray, -1)

        total = self.dash_len + self.dash_gap
        for x in self.boundaries[1:-1]:
            y0 = -self.dash_len + self.road_offset
            while y0 < h:
                pt1 = (int(x - self.dash_w/2), int(y0))
                pt2 = (int(x + self.dash_w/2), int(y0 + self.dash_len))
                cv2.rectangle(overlay, pt1, pt2, self.dash_col, -1)
                y0 += total

        for x in (self.boundaries[0], self.boundaries[-1]):
            cv2.line(overlay, (x,0), (x,h), self.dash_col, self.boundary_th)

        cv2.addWeighted(overlay, self.road_alpha, frame, 1-self.road_alpha, 0, frame)

        for coin in self.coins:
            coin.draw(frame)
        for obs in self.obstacles:
            obs.draw(frame)
        
        #     fr.draw(frame)
        for ht in self.hearts:
            ht.draw(frame)
        
        for effect in self.explosion_effects:
            effect.draw(frame)

        self.car.draw(frame)

        # --- Draw Hitboxes if in Debug Mode ---
        if self.debug_mode:
            self._draw_debug_hitboxes(frame)

        for i in range(self.lives):
            cv2.circle(frame, (w - 50 - i * 40, 50), 15, (0, 0, 255), -1)

        cv2.putText(frame, f"SCORE: {self.score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"COINS: {self.coin_count}", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # --- NEW: Pattern text is now part of debug mode ---
        if self.debug_mode:
            if self.pattern_active:
                cv2.putText(frame, f"PATTERN: {self.pattern_name.upper()}", (10, h - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            elif self.pattern_cooldown_active and not self.game_over:
                remaining = max(0, self.pattern_cooldown_end_time - time.time())
                cv2.putText(frame, f"COOLDOWN... {remaining:.0f}s", (10, h - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    def draw_intro(self, frame):
        h, w = frame.shape[:2]
        cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0, frame)

        cv2.putText(frame, "BLINK 'N DRIFT", (w // 2 - 230, h // 2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, "Pinch to Start", (w // 2 - 180, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, "If you blink, you're cooked!", (w // 2 - 260, h // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 3, cv2.LINE_AA)

    def draw_game_over(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "GAME OVER", (w // 2 - 200, h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, f"SCORE: {self.score}", (w // 2 - 150, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"COINS: {self.coin_count}", (w // 2 - 150, h // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"HIGH SCORE: {self.high_score}", (w // 2 - 180, h // 2 + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "Pinch to Restart", (w // 2 - 220, h // 2 + 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    def reset(self):
        self.obstacles = []
        self.coins = []

        self.hearts = []
        self.explosion_effects = []
        self.lives = MAX_LIVES
        self.score = 0
        self.coin_count = 0
        self.spawn_interval = SPAWN_INTERVAL
        
        self.obstacle_speed = INITIAL_OBSTACLE_SPEED
        
        # --- NEW: Reset pattern engine to start with a cooldown ---
        self.pattern_active = False
        self.pattern_name = None
        self.pattern_step = 0
        self.pattern_end_time = 0
        self.pattern_cooldown_active = True
        self.pattern_cooldown_end_time = time.time() + PATTERN_COOLDOWN_S
        
        t = time.time()
        self.t_last_obs_spawn = t 
        self.t_last_coin_spawn = t

        self.t_last_heart_spawn = t
        self.game_over = False
        self.last_spawn_was_truck = False

    @staticmethod
    def _intersect(car_rect, obs_rect, forgive_tail=True):
        """
        Definitive intersection method using symmetrical, centered core hitboxes.
        Both car and obstacle hitboxes are shrunk equally from the top and bottom.
        """
        ax1, ay1, ax2, ay2 = car_rect
        bx1, by1, bx2, by2 = obs_rect

        # 1. Broad-phase check: If visual boxes don't overlap, no collision.
        if ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2:
            return False

        # For non-forgiving items (coins, hearts), a simple visual overlap is enough.
        if not forgive_tail:
            return True

        # 2. Define Symmetrical Core Hitboxes
        car_w, car_h = ax2 - ax1, ay2 - ay1
        obs_w, obs_h = bx2 - bx1, by2 - by1

        # Shrink car hitbox from all sides
        car_h_shrink = (car_w * CAR_H_FORGIVENESS) / 2
        car_v_shrink = (car_h * CAR_V_FORGIVENESS) / 2 # Shrink from top AND bottom
        core_ax1 = ax1 + car_h_shrink
        core_ay1 = ay1 + car_v_shrink
        core_ax2 = ax2 - car_h_shrink
        core_ay2 = ay2 - car_v_shrink

        # Shrink obstacle hitbox from all sides
        obs_h_shrink = (obs_w * OBS_H_FORGIVENESS) / 2
        obs_v_shrink = (obs_h * OBS_V_FORGIVENESS) / 2 # Shrink from top AND bottom
        core_bx1 = bx1 + obs_h_shrink
        core_by1 = by1 + obs_v_shrink
        core_bx2 = bx2 - obs_h_shrink
        core_by2 = by2 - obs_v_shrink

        # 3. Final check: A collision only occurs if the core hitboxes intersect.
        if (core_ax2 <= core_bx1 or core_ax1 >= core_bx2 or
            core_ay2 <= core_by1 or core_ay1 >= core_by2):
            return False  # No collision between core hitboxes.

        return True

    def is_position_safe(self, y_pos, lane_to_check, item_list, vertical_gap):
        """Checks if a given lane is clear of items in a list within a vertical gap."""
        for item in item_list:
            if item.lane == lane_to_check and abs(item.y - y_pos) < vertical_gap:
                return False
        return True

    def _is_lane_safe(self, y_pos, lane):
        is_safe_from_obstacles = self.is_position_safe(y_pos, lane, self.obstacles, MIN_VERTICAL_GAP)
        is_safe_from_collectibles = self.is_position_safe(y_pos, lane, self.coins + self.hearts, MIN_VERTICAL_GAP)
        return is_safe_from_obstacles and is_safe_from_collectibles


