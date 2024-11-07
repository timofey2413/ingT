import cv2
import mediapipe as mp

def get_face_landmarks(image, draw=False, static_image_mode=True):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,

    )
    image_rows, image_cols, _ = image.shape
    result = face_mesh.process(image_rgb)
    image_landmarks = []

    if result.multi_face_landmarks:
        if draw:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=result.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        single_face = result.multi_face_landmarks[0].landmark
        xs = []
        ys = []
        zs = []

        for i in single_face:
            xs.append(i.x)
            ys.append(i.y)
            zs.append(i.z)

        for j in range(len(xs)):
            image_landmarks.append(xs[j] - min(xs))
            image_landmarks.append(ys[j] - min(ys))
            image_landmarks.append(zs[j] - min(zs))

    return image_landmarks