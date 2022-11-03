from asyncio.windows_events import NULL
import os
from unittest import result
import cv2
import keras
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

root_dir = os.getcwd()

# ------------- Model for face detection---------#
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

model_path = 'model.h5'
model = keras.models.load_model("model.h5")
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


Boxes = [] # to store the face co-ordinates
mssg = 'Face Detected' # to display on image

class Analyse(VideoTransformerBase):
    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        print(faces)
        # If no faces our detected
        # if not faces:
        #     msg = 'No face detected'
        #     cv2.putText(frame, f'{msg}', (40, 40),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (200), 2)
        # else:
            # --------- Bounding Face ---------#
            # for face in faces:
            #     x = face.left() # extracting the face coordinates
            #     y = face.top()
            #     x2 = face.right()
            #     y2 = face.bottom()
            #     # for box in Boxes:
        #     face = frame[y:y2, x:x2]
        for (x,y,w,h) in faces:  
            face = frame[y-5:y+h+5,x-5:x+w+5]
            x2 = x + w
            y2 = y + h
            # ----- Image preprocessing --------#
            resized_face = cv2.resize(face,(100,100))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)
            preds = model.predict(resized_face)[0]
            if preds > 0.5:
                result = 'Pain'
            else: result = 'No-pain'
            cv2.putText(frame, f'{mssg}:{result}', (x,y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x2, y2),
                        (00, 200, 200), 2)
        return frame



def main():
    st.title("Pain Management")
    st.subheader ("Please press the start button and hold on for about 15 seconds...")
    st.subheader("Come closer to the camera and stay still...")
    RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
    webrtc_ctx = webrtc_streamer(
            key="WYH",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory= Analyse,
            async_processing=True,
        )

if __name__ == "__main__":
    main()
