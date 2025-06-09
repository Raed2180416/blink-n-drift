# main.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import mediapipe as mp
import time
import math

from game_logic import Game

# ── Config ────────────────────────────────────────────────────────
EAR_THRESH    = 2.144
EAR_RELEASE   = 2.194
BLINK_FRAMES  = 2
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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
    draw  = mp.solutions.drawing_utils
    return hands, fmesh, draw, mp.solutions.hands


def compute_ear(lm, idxs):
    p = [lm[i] for i in idxs]
    v1 = math.hypot(p[1].x - p[5].x, p[1].y - p[5].y)
    v2 = math.hypot(p[2].x - p[4].x, p[2].y - p[4].y)
    h  = math.hypot(p[0].x - p[3].x, p[0].y - p[3].y)
    return (v1 + v2) / (2.0 * h)


def process_blinks(rgb, fmesh, state, cnt):
    """State-machine blink; OPEN→CLOSED triggers blink_event."""
    blink_event = False
    res = fmesh.process(rgb)
    if res.multi_face_landmarks:
        lm  = res.multi_face_landmarks[0].landmark
        ear = (compute_ear(lm, LEFT_EYE_IDS) +
               compute_ear(lm, RIGHT_EYE_IDS)) / 2.0

        if state == "OPEN" and ear < EAR_THRESH:
            cnt += 1
            if cnt >= BLINK_FRAMES:
                state, cnt, blink_event = "CLOSED", 0, True
        elif state == "OPEN":
            cnt = 0
        elif state == "CLOSED" and ear > EAR_RELEASE:
            state, cnt = "OPEN", 0

    return state, cnt, blink_event


def detect_pinch_and_lane(rgb, hands, pstate, lane, last_sw):
    """Detects thumb-index pinch + index-finger lane choice."""
    pevt = False
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        hlm = res.multi_hand_landmarks[0]

        d = math.hypot(
            hlm.landmark[4].x - hlm.landmark[8].x,
            hlm.landmark[4].y - hlm.landmark[8].y
        )
        if pstate == "OPEN" and d < PINCH_THRESH:
            pstate, pevt = "CLOSED", True
        elif pstate == "CLOSED" and d > PINCH_RELEASE:
            pstate = "OPEN"

        new_lane = max(0, min(LANE_COUNT - 1,
                              int(hlm.landmark[8].x * LANE_COUNT)))
        now = time.time()
        if new_lane != lane and now - last_sw > SWITCH_DELAY:
            lane, last_sw = new_lane, now
            print(f">>> Switched to lane {lane}", flush=True)

    return pstate, pevt, lane, last_sw


def draw_start(frame, fw, fh):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    lines = [
        "BLINK & DRIFT",
        " Drag palm across to switch lanes",
        " Try not to blink, or you might regret it",
        " Collect blue circles to slow down",
        " Collect hearts to regain life",
        "Pinch thumb+index to start, pause, resume, or restart"
    ]
    y0 = int(fh * 0.2)
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt,
                    (int(fw * 0.1), y0 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Camera", frame)


def draw_play(frame, game, lane, fw, fh):
    game.update(lane, fw, fh)
    game.draw(frame)
    cv2.imshow("Camera", frame)


def draw_gameover(frame, game, fw, fh):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    game.save_highscore()
    cv2.putText(frame, f"SCORE: {game.score}",
                (fw//4, fh//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    cv2.putText(frame, f"HIGHSCORE: {game.high_score}",
                (fw//4, fh//2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    cv2.putText(frame, "Pinch to RESTART",
                (fw//4, fh//2 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Camera", frame)


def draw_pause(frame, fw, fh):
    """Simple paused screen."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "PAUSED",
                (fw//3, fh//2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    cv2.putText(frame, "Pinch to resume",
                (fw//3, fh//2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Camera", frame)


def handle_display_and_input():
    k = cv2.waitKey(1) & 0xFF
    return not (k == ord('q') or
                cv2.getWindowProperty("Camera",
                                      cv2.WND_PROP_VISIBLE) < 1)


def handle_pause_toggle(state, pevt):
    """Handle pause/resume toggle logic."""
    if pevt:
        if state == "playing":
            return "paused"
        elif state == "paused":
            return "playing"
    return state


def handle_start_state(frame, fw, fh, pevt):
    """Handle start state logic."""
    draw_start(frame, fw, fh)
    if pevt:
        return "playing", Game(LANE_COUNT)
    return "start", None


def handle_playing_state(frame, game, lane, fw, fh):
    """Handle playing state logic."""
    draw_play(frame, game, lane, fw, fh)
    if game.lives <= 0:
        return "gameover"
    return "playing"


def handle_gameover_state(frame, game, fw, fh, pevt):
    """Handle gameover state logic."""
    draw_gameover(frame, game, fw, fh)
    if pevt:
        return "playing", Game(LANE_COUNT), "OPEN", 0, "OPEN", 1, 0.0
    return "gameover", game, None, None, None, None, None


def process_frame_input(cap, hands, fmesh, blink_st, blink_ctr, pinch_st, lane, last_sw, game, state):
    """Process camera frame and detect input events."""
    ret, frame = cap.read()
    if not ret:
        return None, None, None, None, None, None, None, None, None
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Blink detection
    blink_st, blink_ctr, bevt = process_blinks(rgb, fmesh, blink_st, blink_ctr)
    if bevt and state == "playing":
        game.on_blink()

    # Pinch detection
    pinch_st, pevt, lane, last_sw = detect_pinch_and_lane(
        rgb, hands, pinch_st, lane, last_sw)

    return frame, rgb, bevt, pevt, blink_st, blink_ctr, pinch_st, lane, last_sw


def handle_game_state_transitions(state, game, frame, fw, fh, pevt, blink_st, blink_ctr, pinch_st, lane, last_sw):
    """Handle transitions between game states."""
    if state == "start":
        state, new_game = handle_start_state(frame, fw, fh, pevt)
        if new_game:
            game = new_game
    elif state == "playing":
        state = handle_playing_state(frame, game, lane, fw, fh)
    else:  # gameover
        result = handle_gameover_state(frame, game, fw, fh, pevt)
        state, game, blink_st, blink_ctr, pinch_st, lane, last_sw = (
            result[0], result[1],
            result[2] or blink_st, result[3] or blink_ctr,
            result[4] or pinch_st, result[5] or lane, result[6] or last_sw
        )
    
    return state, game, blink_st, blink_ctr, pinch_st, lane, last_sw


def run_game_loop(cap, fw, fh, hands, fmesh):
    state = "start"
    lane = 1
    last_sw = 0.0
    blink_st = "OPEN"
    blink_ctr = 0
    pinch_st = "OPEN"
    game = Game(LANE_COUNT)

    while True:
        # Process frame and input
        frame_data = process_frame_input(cap, hands, fmesh, blink_st, blink_ctr, 
                                       pinch_st, lane, last_sw, game, state)
        if frame_data[0] is None:
            break
        
        frame, _, _, pevt, blink_st, blink_ctr, pinch_st, lane, last_sw = frame_data

        # Handle pause toggle
        new_state = handle_pause_toggle(state, pevt)
        if new_state != state:
            state = new_state
            continue

        # Handle paused state
        if state == "paused":
            draw_pause(frame, fw, fh)
            if not handle_display_and_input():
                break
            continue

        # Handle game state transitions
        state, game, blink_st, blink_ctr, pinch_st, lane, last_sw = handle_game_state_transitions(
            state, game, frame, fw, fh, pevt, blink_st, blink_ctr, pinch_st, lane, last_sw)

        if not handle_display_and_input():
            break


def main():
    cap = setup_camera()
    if not cap:
        return

    ret, first = cap.read()
    if not ret:
        print("ERROR: Could not read initial frame."); return
    first = cv2.flip(first, 1)
    fh, fw = first.shape[:2]

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Camera", fw, fh)

    hands, fmesh, _, _ = init_mediapipe()
    run_game_loop(cap, fw, fh, hands, fmesh)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
