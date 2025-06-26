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
COIN_INTERVAL = 3.0
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
PATTERN_DURATION = 30
BONUS_DURATION = 10


OBSTACLE_PATTERNS = {
    "zigzag": [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ],
    "wall_gap": [
        [1, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    "alternating": [
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
    ],
    "funnel": [
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ],
    "spiral": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
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

        self.pattern_active = False
        self.pattern_name = None
        self.pattern_step = 0
        self.pattern_spawn_timer = 0
        self.pattern_start_score = 0
        
        self.bonus_active = False
        self.bonus_start_time = 0
        self.original_spawn_interval = self.spawn_interval

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
        # REMOVED: No longer need to set global variable

    def on_blink(self):
        old_interval = self.spawn_interval
        old_speed = self.obstacle_speed
        
        self.spawn_interval = max(0.5, self.spawn_interval * 0.9)
        self.obstacle_speed += SPEED_INCREMENT # REMOVED: Speed cap is gone.
        
        print(f">>> Blink! Interval: {old_interval:.2f}s→{self.spawn_interval:.2f}s, Speed: {old_speed}→{self.obstacle_speed}")

    def is_position_safe(self, y_pos, lane_to_check, item_list):
        for item in item_list:
            if item.lane == lane_to_check and abs(item.y - y_pos) < MIN_VERTICAL_GAP:
                return False
        return True

    def _should_start_pattern(self):
        if self.pattern_active or self.bonus_active:
            return False
            
        # Check for bonus mode at multiples of 70
        if self.score > 0 and self.score % 70 == 0:
            return "bonus"
            
        # Check for pattern mode (25-35 range in each 50-point cycle, starting from score 25)
        if self.score >= 25:
            # Calculate position within the 70-point cycle
            cycle_position = (self.score - 25) % 70
            
            # Pattern mode occurs at positions 0-10 within each 70-point cycle
            # (which corresponds to scores 25-35, 95-105, 165-175, etc.)
            if 0 <= cycle_position <= 10:
                # Only start pattern at the beginning of the range
                if cycle_position == 0:
                    return "pattern"
                
        return False

    def _start_pattern_mode(self):
        self.pattern_active = True
        self.pattern_step = 0
        self.pattern_spawn_timer = 0
        self.pattern_start_score = self.score
        
        patterns = list(OBSTACLE_PATTERNS.keys())
        pattern_index = (self.score // 25) % len(patterns)
        self.pattern_name = patterns[pattern_index]
        
        print(f">>> Starting pattern '{self.pattern_name}' at score {self.score}")

    def _start_bonus_stretch(self):
        self.bonus_active = True
        self.bonus_start_time = time.time()
        self.original_spawn_interval = self.spawn_interval
        self.spawn_interval = 3.0
        
        print(f">>> Starting bonus stretch at score {self.score}")
        
        self._spawn_bonus_collectibles()

    def _spawn_bonus_collectibles(self):
        for lane in range(self.lane_count):
            if self.coin_sprite is not None:
                coin = CoinItem(self.lane_count, self.coin_sprite)
                coin.lane = lane
                coin.y = SPAWN_Y - (lane * 100)
                self.coins.append(coin)
            
            if lane % 2 == 0 and self.heart_sprite is not None:
                heart = HeartItem(self.lane_count, self.heart_sprite)
                heart.lane = lane
                heart.y = SPAWN_Y - (lane * 100) - 50
                self.hearts.append(heart)

    def _update_pattern_mode(self, now):
        if not self.pattern_active:
            return
            
        if self.score >= self.pattern_start_score + PATTERN_DURATION:
            self.pattern_active = False
            self.pattern_name = None
            print(f">>> Pattern ended at score {self.score}")
            return
            
        pattern = OBSTACLE_PATTERNS[self.pattern_name]
        
        if now - self.pattern_spawn_timer > 0.8:
            current_pattern = pattern[self.pattern_step % len(pattern)]
            
            for lane_idx, should_spawn in enumerate(current_pattern):
                if should_spawn and lane_idx < self.lane_count and self.car_sprites:
                    # Patterns will only spawn regular cars for consistency
                    sprite = random.choice(self.car_sprites)
                    new_obstacle = Obstacle(self.lane_count, sprite)
                    new_obstacle.lane = lane_idx
                    new_obstacle.y = SPAWN_Y - (len([l for l in current_pattern[:lane_idx] if l]) * 80)
                    self.obstacles.append(new_obstacle)
            
            self.pattern_step += 1
            self.pattern_spawn_timer = now

    def _update_bonus_stretch(self, now):
        if not self.bonus_active:
            return
            
        if now - self.bonus_start_time > BONUS_DURATION:
            self.bonus_active = False
            self.spawn_interval = self.original_spawn_interval
            print(f">>> Bonus stretch ended at score {self.score}")
            return
            
        # Continue spawning bonus collectibles during bonus time
        if now - self.t_last_coin_spawn > 2.0:
            self._spawn_bonus_collectibles()
            self.t_last_coin_spawn = now

    def _spawn_obstacle(self, now):
        if self.pattern_active or self.bonus_active:
            return
            
        if not self._can_spawn_obstacle(now):
            return

        lookahead_distance = self.obstacle_speed * OBSTACLE_SPAWN_LOOKAHEAD_SECONDS
        avg_lane_width = (self.boundaries[1] - self.boundaries[0])
        avg_obs_visual_height = OBS_SCALE * avg_lane_width * 1.7

        candidate_lanes = self._evaluate_candidate_lanes(lookahead_distance, avg_obs_visual_height)
        chosen_lane = self._choose_spawn_lane(candidate_lanes)

        if chosen_lane != -1:
            self._create_and_spawn_obstacle(chosen_lane, now)

    def _can_spawn_obstacle(self, now):
        if not (self.car_sprites or self.truck_sprites) or now - self.t_last_obs_spawn < self.spawn_interval:
            return False
        
        return (self.boundaries and len(self.boundaries) > 1 and self.boundaries[1] > self.boundaries[0])

    def _evaluate_candidate_lanes(self, lookahead_distance, avg_obs_visual_height):
        candidate_lanes = []
        
        for lane_idx in range(self.lane_count):
            if not self._is_lane_basically_safe(lane_idx):
                continue
                
            safety_score = self._calculate_lane_safety_score(lane_idx, lookahead_distance, avg_obs_visual_height)
            if safety_score > 0:
                candidate_lanes.append((lane_idx, safety_score))
        
        return candidate_lanes

    def _is_lane_basically_safe(self, lane_idx):
        return (self.is_position_safe(SPAWN_Y, lane_idx, self.obstacles) and 
                self.is_position_safe(SPAWN_Y, lane_idx, self.coins + self.hearts))

    def _calculate_lane_safety_score(self, lane_idx, lookahead_distance, avg_obs_visual_height):
        occupied_lanes = self._get_occupied_critical_lanes(lookahead_distance, avg_obs_visual_height)
        potential_lanes_occupied = list(occupied_lanes)
        potential_lanes_occupied[lane_idx] = True
        
        if all(potential_lanes_occupied):
            return 0
        
        safety_score = self._calculate_base_safety_score(potential_lanes_occupied)
        safety_score += self._calculate_adjacency_bonus(lane_idx, occupied_lanes)
        
        return safety_score

    def _get_occupied_critical_lanes(self, lookahead_distance, avg_obs_visual_height):
        occupied_lanes = [False] * self.lane_count
        
        for obs in self.obstacles:
            if self._is_obstacle_critical(obs, lookahead_distance, avg_obs_visual_height):
                if 0 <= obs.lane < self.lane_count:
                    occupied_lanes[obs.lane] = True
        
        return occupied_lanes

    def _is_obstacle_critical(self, obs, lookahead_distance, avg_obs_visual_height):
        min_y = SPAWN_Y - avg_obs_visual_height * 0.75
        max_y = SPAWN_Y + lookahead_distance
        return min_y < obs.y < max_y

    def _calculate_base_safety_score(self, potential_lanes_occupied):
        blocked_count = sum(1 for occupied in potential_lanes_occupied if occupied)
        return (self.lane_count - blocked_count) * 2

    def _calculate_adjacency_bonus(self, lane_idx, occupied_lanes):
        if self.lane_count <= 1:
            return 0
        
        is_adjacent = self._is_adjacent_to_occupied_lane(lane_idx, occupied_lanes)
        return 0 if is_adjacent else 1

    def _is_adjacent_to_occupied_lane(self, lane_idx, occupied_lanes):
        if lane_idx > 0 and occupied_lanes[lane_idx - 1]:
            return True
        if lane_idx < self.lane_count - 1 and occupied_lanes[lane_idx + 1]:
            return True
        return False

    def _choose_spawn_lane(self, candidate_lanes):
        if not candidate_lanes:
            return -1 # CRITICAL FIX: If no safe lane is found, do not spawn.

        # The rest of the logic is fine, it picks the best from the safe candidates.
        candidate_lanes.sort(key=lambda x: x[1], reverse=True)
        best_score = candidate_lanes[0][1]
        top_tier_lanes = [lane_idx for lane_idx, score in candidate_lanes if score == best_score]
        return random.choice(top_tier_lanes)

    def _create_and_spawn_obstacle(self, chosen_lane, now):
        sprite = None
        # --- NEW: Logic for spawning rare trucks ---
        # Trucks have a ~20% chance to spawn, but not if the last obstacle was also a truck.
        is_truck_spawn = random.random() < 0.20 and not self.last_spawn_was_truck

        if is_truck_spawn and self.truck_sprites:
            sprite = random.choice(self.truck_sprites)
            self.last_spawn_was_truck = True
        elif self.car_sprites:
            sprite = random.choice(self.car_sprites)
            self.last_spawn_was_truck = False

        if sprite is not None:
            new_obstacle = Obstacle(self.lane_count, sprite)
            new_obstacle.lane = chosen_lane
            self.obstacles.append(new_obstacle)
            self.t_last_obs_spawn = now
    
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
            mode_check = self._should_start_pattern()
            if mode_check == "pattern":
                self._start_pattern_mode()
            elif mode_check == "bonus":
                self._start_bonus_stretch()
            
            self._update_pattern_mode(now)
            self._update_bonus_stretch(now)
            
            self._spawn_obstacle(now)
            
            if not self.bonus_active:
                self._spawn_coin(now)
                self._spawn_heart(now)
            
            # self._spawn_freeze(now) # Removed

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

        if self.pattern_active:
            cv2.putText(frame, f"PATTERN: {self.pattern_name.upper()}", (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        elif self.bonus_active:
            remaining = max(0, BONUS_DURATION - (time.time() - self.bonus_start_time))
            cv2.putText(frame, f"BONUS TIME! {remaining:.0f}s", (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    def draw_intro(self, frame):
        h, w = frame.shape[:2]
        dark_overlay = frame.copy()
        cv2.rectangle(dark_overlay, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(dark_overlay, 0.7, frame, 0.3, 0, frame)

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
        # REMOVED: No longer need to set global variable
        
        self.pattern_active = False
        self.pattern_name = None
        self.pattern_step = 0
        self.pattern_spawn_timer = 0
        self.pattern_start_score = 0
        
        self.bonus_active = False
        self.bonus_start_time = 0
        self.spawn_interval = SPAWN_INTERVAL
        self.original_spawn_interval = SPAWN_INTERVAL
        
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

    def _is_lane_safe(self, y_pos, lane):
        is_safe_from_obstacles = self.is_position_safe(y_pos, lane, self.obstacles)
        is_safe_from_collectibles = self.is_position_safe(y_pos, lane, self.coins + self.hearts)
        return is_safe_from_obstacles and is_safe_from_collectibles


 
