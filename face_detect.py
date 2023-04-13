import streamlit as st
import numpy as np
import cv2 as cv

st.subheader('Phát hiện khuôn mặt')
FRAME_WINDOW = st.image([])
deviceId = 0
cap = cv.VideoCapture(deviceId)

if not cap.isOpened():
    st.error('Không thể kết nối tới camera. Vui lòng kiểm tra lại!')

else:
    st.success('Đã kết nối tới camera!')

    def visualize(input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):

                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
        cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    detector = cv.FaceDetectorYN.create(
        'face_detection_yunet_2022mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )

    tm = cv.TickMeter()
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        frame = cv.resize(frame, (frameWidth, frameHeight))

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        FRAME_WINDOW.image(frame, channels='BGR')

        if st.button('Stop'):
            break

    cap.release()
    st.success('Camera đã ngừng kết nối!') 
