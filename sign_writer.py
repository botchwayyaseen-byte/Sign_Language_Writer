"""
sign_writer.py - fingerspelling -> on-screen text, in one file.

Setup:
    pip install opencv-python mediapipe scikit-learn numpy joblib

Usage:
    python sign_writer.py collect   # record your own samples for each sign
    python sign_writer.py train     # train the classifier
    python sign_writer.py run       # live: sign letters, sentence appears on screen

Collect-mode keys:
    a-z      record a burst of samples for that letter
    1        record the "space" gesture (e.g. flat open palm)
    2        record the "delete" gesture (e.g. closed fist / thumb down)
    q        quit
Tips: for each sign, record 2-3 bursts, moving your hand slightly (distance,
angle, left/right position). Do NOT include a letter you don't need.
Works for any sign language's static handshapes (ASL, TID, ...): the labels
are just the keys you press, so record the alphabet of your choice.

Run-mode keys:
    c  clear the sentence      b  backspace      s  save sentence to sentences.txt
    q  quit
"""
import sys
import time
import csv
import os
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp

DATA_FILE = "samples.csv"
MODEL_FILE = "sign_model.joblib"
BURST = 40                 # samples recorded per key press
STABLE_FRAMES = 12         # same prediction for N frames before accepting
MIN_CONF = 0.75            # minimum classifier probability
SPECIAL = {"1": "space", "2": "del"}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ------------------------------------------------------------- sci-fi HUD
FONT = cv2.FONT_HERSHEY_DUPLEX
CYAN = (255, 255, 0)        # BGR
WHITE = (255, 255, 255)
GREEN = (120, 255, 80)
RED = (60, 60, 255)


def spaced_width(text, scale, thick, spacing):
    return sum(cv2.getTextSize(c, FONT, scale, thick)[0][0] + spacing for c in text)


def spaced_text(img, text, x, y, scale, color, thick=1, spacing=6, last_color=None):
    """Letter-spaced text; returns the x position after the last character."""
    for i, c in enumerate(text):
        col = last_color if (last_color and i == len(text) - 1) else color
        cv2.putText(img, c, (x, y), FONT, scale, col, thick, cv2.LINE_AA)
        x += cv2.getTextSize(c, FONT, scale, thick)[0][0] + spacing
    return x


def corner_brackets(img, x1, y1, x2, y2, color, length=18, thick=2):
    for x, y, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (x, y), (x + dx * length, y), color, thick, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * length), color, thick, cv2.LINE_AA)


def draw_hud(frame, sentence, current, conf, hand_present):
    h, w = frame.shape[:2]
    t = time.time()

    # subtle scanlines
    frame[::3] = (frame[::3] * 0.8).astype(np.uint8)

    # translucent bottom panel with corner brackets
    x1, y1, x2, y2 = 15, h - 100, w - 15, h - 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (35, 15, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    corner_brackets(frame, x1, y1, x2, y2, CYAN)
    cv2.line(frame, (x1 + 20, y1 + 28), (x2 - 20, y1 + 28), (140, 110, 0), 1, cv2.LINE_AA)

    # sentence, trimmed from the left so it always fits the panel
    scale, sp = 0.9, 6
    shown = sentence.upper()
    while shown and spaced_width(shown, scale, 1, sp) > (x2 - x1) - 70:
        shown = shown[1:]
    tx, ty = x1 + 25, y2 - 18
    blink = int(t * 2) % 2 == 0

    # draw once on a glow layer (thick) and once crisp, then blur + add the glow
    glow = np.zeros_like(frame)
    for img, thick, last in ((glow, 3, CYAN), (frame, 1, WHITE)):
        cv2.putText(img, "> INPUT STREAM", (x1 + 25, y1 + 20), FONT, 0.45, CYAN, 1, cv2.LINE_AA)
        end_x = spaced_text(img, shown, tx, ty, scale, CYAN, thick, sp, last)
        if blink:
            cv2.rectangle(img, (end_x, ty - 22), (end_x + 12, ty + 2), CYAN, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 7)
    cv2.add(frame, glow, frame)

    # top-left status readout
    col = GREEN if hand_present else RED
    cv2.circle(frame, (28, 28), 6, col, -1, cv2.LINE_AA)
    cv2.putText(frame, "HAND LOCK" if hand_present else "NO SIGNAL", (42, 33), FONT, 0.5, col, 1, cv2.LINE_AA)
    if current:
        cv2.putText(frame, "DETECTED", (20, 62), FONT, 0.45, CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, current.upper(), (20, 108), FONT, 1.4, WHITE, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (20, 120), (180, 128), CYAN, 1)
        cv2.rectangle(frame, (20, 120), (20 + int(160 * conf), 128), CYAN, -1)
        cv2.putText(frame, f"{conf:.0%}", (190, 128), FONT, 0.45, CYAN, 1, cv2.LINE_AA)


def make_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


def features(hand_landmarks):
    """Wrist-relative, scale-normalised 63-dim vector."""
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    pts -= pts[0]                       # translate: wrist at origin
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale < 1e-6:
        return None
    pts /= scale                        # scale invariance
    return pts.flatten()


def read_hand(frame, hands):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        return res.multi_hand_landmarks[0]
    return None


def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam (index 0).")
    return cap


# ----------------------------------------------------------------- collect
def collect():
    cap, hands = open_camera(), make_hands()
    counts = Counter()
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            for row in csv.reader(f):
                counts[row[0]] += 1

    active_label, remaining = None, 0
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            lm = read_hand(frame, hands)
            if lm:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                if remaining > 0:
                    feat = features(lm)
                    if feat is not None:
                        writer.writerow([active_label] + feat.tolist())
                        counts[active_label] += 1
                        remaining -= 1

            status = f"recording '{active_label}' {BURST - remaining}/{BURST}" if remaining > 0 else "press a-z / 1 / 2 to record, q to quit"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            summary = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            cv2.putText(frame, summary[-90:], (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("collect", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if remaining == 0 and key != 255:
                ch = chr(key)
                if ch in SPECIAL:
                    active_label, remaining = SPECIAL[ch], BURST
                elif ch.isalpha() and ch.islower():
                    active_label, remaining = ch.upper(), BURST
    cap.release()
    cv2.destroyAllWindows()


# ------------------------------------------------------------------- train
def train():
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    if not os.path.exists(DATA_FILE):
        sys.exit("No samples.csv found. Run 'collect' first.")
    X, y = [], []
    with open(DATA_FILE) as f:
        for row in csv.reader(f):
            y.append(row[0])
            X.append([float(v) for v in row[1:]])
    X, y = np.array(X), np.array(y)
    print(f"{len(y)} samples, {len(set(y))} classes: {sorted(set(y))}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)
    print(classification_report(y_te, clf.predict(X_te)))
    print("Note: frames from the same burst are highly correlated, so this score is optimistic.")
    clf.fit(X, y)                       # final model on all data
    joblib.dump(clf, MODEL_FILE)
    print(f"Saved {MODEL_FILE}")


# --------------------------------------------------------------------- run
def run():
    import joblib
    if not os.path.exists(MODEL_FILE):
        sys.exit("No model found. Run 'train' first.")
    clf = joblib.load(MODEL_FILE)
    cap, hands = open_camera(), make_hands()

    sentence = ""
    history = deque(maxlen=STABLE_FRAMES)
    armed = True            # must drop the hand / change sign before the same letter repeats
    last_accepted = None
    current, conf = "", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        lm = read_hand(frame, hands)

        if lm:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            feat = features(lm)
            if feat is not None:
                probs = clf.predict_proba([feat])[0]
                i = int(np.argmax(probs))
                current, conf = clf.classes_[i], float(probs[i])
                history.append(current if conf >= MIN_CONF else None)

                if len(history) == STABLE_FRAMES and history.count(current) == STABLE_FRAMES:
                    if current != last_accepted or armed:
                        if current == "space":
                            sentence += " "
                        elif current == "del":
                            sentence = sentence[:-1]
                        else:
                            sentence += current
                        last_accepted, armed = current, False
                        history.clear()
        else:
            history.clear()
            armed = True        # hand left the frame -> same letter may repeat
            current, conf = "", 0.0

        # ---- UI
        draw_hud(frame, sentence, current, conf, lm is not None)
        cv2.imshow("sign writer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence = ""
        elif key == ord("b"):
            sentence = sentence[:-1]
        elif key == ord("s") and sentence.strip():
            with open("sentences.txt", "a", encoding="utf-8") as f:
                f.write(sentence.strip() + "\n")
            sentence = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    {"collect": collect, "train": train, "run": run}.get(mode, lambda: print(__doc__))()
