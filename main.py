import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import cv2
import mediapipe as mp
import time
import math
from game_logic1 import Game
# ── Config ────────────────────────────────────────────────────────
EAR_THRESH    = 2.280     # Keep this
EAR_RELEASE   = 2.260     # Keep this
BLINK_FRAMES  = 2.4         # Change from 2.35 to integer 2
LEFT_EYE_IDS  = [33, 246, 161, 160, 159, 158]
RIGHT_EYE_IDS = [362, 398, 382, 381, 380, 373]
LANE_COUNT    = 3
SWITCH_DELAY  = 0.2
PINCH_THRESH  = 0.05
PINCH_RELEASE = 0.10
def setup_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap
def init_mediapipe():
    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    fmesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return hands, fmesh

def compute_ear(lm, idxs):
    p = [lm[i] for i in idxs]
    v1 = math.hypot(p[1].x - p[5].x, p[1].y - p[5].y)
    v2 = math.hypot(p[2].x - p[4].x, p[2].y - p[4].y)
    h  = math.hypot(p[0].x - p[3].x, p[0].y - p[3].y)
    return (v1 + v2) / (2.0 * h) if h != 0 else 0

def process_blinks(rgb, fmesh, state, cnt):
    blink_event = False
    res = fmesh.process(rgb)
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        # Calculate average EAR
        left_ear = compute_ear(lm, LEFT_EYE_IDS)
        right_ear = compute_ear(lm, RIGHT_EYE_IDS)
        ear = (left_ear + right_ear) / 2.0
        # Debug print with clearer state indication
        print(f"EAR: {ear:.3f} | State: {state} | Counter: {cnt} {'[BLINK CHARGING]' if cnt > 0 else ''}", end='\r')
        
        if state == "OPEN" and ear > EAR_THRESH:  # High EAR means eyes are closing
            cnt += 1
            if cnt >= BLINK_FRAMES:
                state = "CLOSED"
                cnt = 0
                blink_event = True
                print(f"\nBLINK DETECTED! EAR: {ear:.3f}")
        elif state == "OPEN":
            cnt = 0
        elif state == "CLOSED" and ear < EAR_RELEASE:  # Low EAR means eyes are open
            state = "OPEN"
            cnt = 0
            
    return state, cnt, blink_event

def detect_pinch_and_lane(rgb, hands, pstate, lane, last_sw):
    pevt = False
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        hlm = res.multi_hand_landmarks[0]
        d = math.hypot(hlm.landmark[4].x - hlm.landmark[8].x,
                       hlm.landmark[4].y - hlm.landmark[8].y)
        if pstate == "OPEN" and d < PINCH_THRESH:
            pstate, pevt = "CLOSED", True
        elif pstate == "CLOSED" and d > PINCH_RELEASE:
            pstate = "OPEN"
        new_lane = max(0, min(LANE_COUNT - 1,
                              int(hlm.landmark[8].x * LANE_COUNT)))
        now = time.time()
        if new_lane != lane and now - last_sw > SWITCH_DELAY:
            lane, last_sw = new_lane, now
    return pstate, pevt, lane, last_sw

def handle_display_and_input():
    k = cv2.waitKey(1) & 0xFF
    return not (k == ord('q') or
                cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1)

def draw_start(frame, fw, fh):
    """Draw the start screen on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    lines = [
        "BLINK & DRIFT",
        " Drag palm across to switch lanes",
        " Try not to blink, or you might regret it",
        " Collect hearts to regain life",
        "Pinch thumb + index to start, pause, resume, or restart"
    ]
    y0 = int(fh * 0.2)
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt,
                    (int(fw * 0.1), y0 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Camera", frame)

def draw_countdown(frame, fw, fh):
    """
    Cleaner countdown with disappearing numbers
    """
    for count in range(3, 0, -1):
        # Clear previous number
        for _ in range(2):  # Show each number and clear it
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            if _ == 0:  # Show number
                cv2.putText(frame, str(count), (fw // 2 - 50, fh // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 10)
            
            cv2.imshow("Camera", frame)
            cv2.waitKey(350)  # Show/clear for 350ms each
    
    # Show GO!
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, "GO!", (fw // 2 - 80, fh // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 8)
    cv2.imshow("Camera", frame)
    cv2.waitKey(500)

# --- Helper functions for game states ---
def handle_intro_state(frame, game, pinch_event, fw, fh):
    game.draw_intro(frame) 
    cv2.imshow("Camera", frame)
    if pinch_event:
        draw_countdown(frame, fw, fh) # Countdown
        game.reset()                  # Game state reset
        return "playing"
    return "intro"

def handle_playing_state(frame, game, pinch_event, blink_event, lane, fw, fh):
    if pinch_event:
        return "paused"
    
    if blink_event: # Handle blink effect
        game.on_blink()
        # Visual feedback for blink can be part of game.draw or added here
        cv2.putText(frame, "BLINK!", (fw // 2 - 100, fh // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)

    game.update(lane, fw, fh)
    if game.game_over: # Check game_over status after update
        return "gameover"
        
    game.draw(frame) # game.draw already handles lives
    cv2.imshow("Camera", frame)
    return "playing"

def handle_paused_state(frame, pinch_event, fw, fh):
    # Draw paused overlay (can be a static image or drawn text)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, "PAUSED", (fw // 2 - 150, fh // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "Pinch to resume", (fw // 2 - 200, fh // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Camera", frame)
    if pinch_event:
        return "playing"
    return "paused"

def handle_gameover_state(frame, game, pinch_event, fw, fh):
    game.draw_game_over(frame)
    cv2.imshow("Camera", frame)
    if pinch_event:
        draw_countdown(frame, fw, fh) # Countdown
        game.reset()                  # Game state reset
        return "playing"
    return "gameover"

# ── Main Game Loop ────────────────────────────────────────────────
def run_game_loop(cap, fw, fh, hands, fmesh):
    current_game_state = "intro" # Renamed 'state' to 'current_game_state' for clarity
    lane = LANE_COUNT // 2 # Initial lane
    last_sw = 0.0
    blink_st = "OPEN"
    blink_ctr = 0
    pinch_st = "OPEN"
    game = Game(LANE_COUNT)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Blink detection
        blink_st, blink_ctr, blink_event_flag = process_blinks(rgb, fmesh, blink_st, blink_ctr)
        # Pinch detection and lane switching
        pinch_st, pinch_event_flag, new_lane, last_sw = detect_pinch_and_lane(
            rgb, hands, pinch_st, lane, last_sw)
        lane = new_lane # Update lane based on detection
        # --- Game State Machine ---
        if current_game_state == "intro":
            current_game_state = handle_intro_state(frame, game, pinch_event_flag, fw, fh)
        
        elif current_game_state == "playing":
            current_game_state = handle_playing_state(frame, game, pinch_event_flag, blink_event_flag, lane, fw, fh)
        
        elif current_game_state == "paused":
            current_game_state = handle_paused_state(frame, pinch_event_flag, fw, fh)
            
        elif current_game_state == "gameover":
            current_game_state = handle_gameover_state(frame, game, pinch_event_flag, fw, fh)
        if not handle_display_and_input():
            break
    # Release resources if loop breaks
    hands.close()
    fmesh.close()

def main():
    cap = setup_camera()
    if not cap:
        return

    ret, first = cap.read()
    if not ret:
        print("ERROR: Couldn't read initial frame.")
        return
    first = cv2.flip(first, 1)
    fh, fw = first.shape[:2]

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Camera", fw, fh)

    hands, fmesh = init_mediapipe()
    try:
        run_game_loop(cap, fw, fh, hands, fmesh)
    finally:
    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
