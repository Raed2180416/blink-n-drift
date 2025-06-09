import cv2, mediapipe as mp, time, math

# Same indices as before
LEFT_EYE_IDS  = [33, 246, 161, 160, 159, 158]
RIGHT_EYE_IDS = [362, 398, 382, 381, 380, 373]

def compute_ear(lm, ids):
    p1, p2, p3, p4, p5, p6 = [lm[i] for i in ids]
    v1 = math.hypot(p2.x-p6.x, p2.y-p6.y)
    v2 = math.hypot(p3.x-p5.x, p3.y-p5.y)
    h  = math.hypot(p1.x-p4.x, p1.y-p4.y)
    return (v1+v2)/(2*h)

def sample_ear(duration=5):
    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    start = time.time()
    ears = []
    print(f"Sampling EAR for {duration} seconds. Keep your eyes OPEN.")
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret: continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            ears.append(compute_ear(lm, LEFT_EYE_IDS))
            ears.append(compute_ear(lm, RIGHT_EYE_IDS))
        cv2.waitKey(1)
    cap.release()
    return ears

if __name__ == "__main__":
    import statistics
    open_ears = sample_ear(5)
    print("Open EAR: min {:.3f}, max {:.3f}, mean {:.3f}".format(
        min(open_ears), max(open_ears), statistics.mean(open_ears)
    ))

    input("Now close your eyes firmly and press Enter...")
    closed_ears = sample_ear(3)
    print("Closed EAR: min {:.3f}, max {:.3f}, mean {:.3f}".format(
        min(closed_ears), max(closed_ears), statistics.mean(closed_ears)
    ))

    # Suggest thresholds
    thresh = (statistics.mean(open_ears) + statistics.mean(closed_ears)) / 2
    release = thresh + 0.05
    print(f"\nSuggested EAR_THRESH ≈ {thresh:.3f}")
    print(f"Suggested EAR_RELEASE ≈ {release:.3f}")
