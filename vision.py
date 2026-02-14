import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# Stability buffer
emotion_buffer = deque(maxlen=10)

# Create FaceLandmarker
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1,
)

detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)


def classify_emotion(blendshapes):
    emotion = "neutral"
    confidence = 0.0

    smile = (
        blendshapes.get("mouthSmileLeft", 0) + blendshapes.get("mouthSmileRight", 0)
    ) / 2

    jaw_open = blendshapes.get("jawOpen", 0)
    brow_up = blendshapes.get("browInnerUp", 0)
    frown = (
        blendshapes.get("mouthFrownLeft", 0) + blendshapes.get("mouthFrownRight", 0)
    ) / 2

    if smile > 0.6:
        emotion = "happy"
        confidence = smile
    elif jaw_open > 0.6 and brow_up > 0.3:
        emotion = "shocked"
        confidence = jaw_open
    elif brow_up > 0.5:
        emotion = "surprised"
        confidence = brow_up
    elif frown > 0.5:
        emotion = "sad"
        confidence = frown

    return emotion, confidence


while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    emotion = "neutral"
    confidence = 0.0

    if result.face_blendshapes:
        blendshapes = {b.category_name: b.score for b in result.face_blendshapes[0]}

    emotion, confidence = classify_emotion(blendshapes)

    # Add to stability buffer
    emotion_buffer.append(emotion)
    stable_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

    # Display emotion
    cv2.putText(
        frame,
        stable_emotion,
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # Show frame AFTER drawing text
    cv2.imshow("Camera", frame)

    # Structured state output
    state = {
        "emotion": stable_emotion,
        "confidence": round(confidence, 2),
    }

    print(state)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
