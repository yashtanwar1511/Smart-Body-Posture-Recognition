import cv2
import mediapipe as mp
import time

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/pose_landmarker_lite.task"
    )
)

landmarker = PoseLandmarker.create_from_options(options)

# Camera
cap = cv2.VideoCapture(0)

# 3-second stability timer
correct_start_time = None
READY_TIME = 3

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera open nahi hua")
        break

    # Convert image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect pose
    result = landmarker.detect(mp_image)

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # Left and right shoulder
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        # Check shoulder alignment
        shoulder_difference = abs(
            left_shoulder.y - right_shoulder.y
        )

        if shoulder_difference < 0.05:

            status = "POSTURE CORRECT"
            color = (0, 255, 0)

            # Start timer
            if correct_start_time is None:
                correct_start_time = time.time()

            elapsed_time = time.time() - correct_start_time

            # After 3 seconds
            if elapsed_time >= READY_TIME:

                status = "READY"
                guidance = "Measurement can start"

            else:

                remaining = READY_TIME - elapsed_time
                guidance = f"Hold position: {remaining:.1f} sec"

        else:

            status = "POSTURE INCORRECT"
            guidance = "Keep your shoulders level"
            color = (0, 0, 255)

            # Reset timer
            correct_start_time = None

        # Draw pose points
        height, width, _ = frame.shape

        for landmark in landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # Show status
        cv2.putText(
            frame,
            status,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        # Show guidance
        cv2.putText(
            frame,
            guidance,
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    cv2.imshow(
        "Smart Body Posture Recognition",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()