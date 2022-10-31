from asyncio.windows_events import NULL
import os
import cv2
import dlib
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# ------------- Model for face detection---------#
face_detector = dlib.get_frontal_face_detector()
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

Boxes = [] # to store the face co-ordinates
mssg = 'Face Detected' # to display on image

class Analyse(VideoTransformerBase):
    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        # If no faces our detected
        if not faces:
            msg = 'No face detected'
            cv2.putText(frame, f'{msg}', (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (200), 2)
        else:
            # --------- Bounding Face ---------#
            for face in faces:
                x = face.left() # extracting the face coordinates
                y = face.top()
                x2 = face.right()
                y2 = face.bottom()
                # for box in Boxes:
                face = frame[y:y2, x:x2]
                # ----- Image preprocessing --------#
                blob = cv2.dnn.blobFromImage(
                    face, 1.0, (227, 227), model_mean, swapRB=False)

                # # -------Age Prediction---------#
                # age_Net.setInput(blob)
                # age_preds = age_Net.forward()
                # print(age_preds)
                # # age = ageList[age_preds[0].argmax()]
                # age = '(8 - 12)'
                # print(age)

                cv2.putText(frame, f'{mssg}', (x,y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x2, y2),
                            (00, 200, 200), 2)
        return frame

# def analyse_video(VideoTransformerBase):
#     age = ''
#     global age_prediction 
#     global spoof_prediction 
#     frame = frame.to_ndarray(format="bgr24")
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces = face_detector(gray)
#     # If no faces our detected
#     if not faces:
#         msg = 'No face detected'
#         cv2.putText(frame, f'{msg}', (40, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (200), 2)
#     else:
#         # --------- Bounding Face ---------#
#         for face in faces:
#             x = face.left() # extracting the face coordinates
#             y = face.top()
#             x2 = face.right()
#             y2 = face.bottom()
#             # for box in Boxes:
#             face = frame[y:y2, x:x2]
#             # ----- Image preprocessing --------#
#             blob = cv2.dnn.blobFromImage(
#                 face, 1.0, (227, 227), model_mean, swapRB=False)

#             # -------Age Prediction---------#
#             age_Net.setInput(blob)
#             age_preds = age_Net.forward()
#             # print(age_preds)
#             age = ageList[age_preds[0].argmax()]
#             age_prediction.append(age)
#             # age = '(21-24)'
#             # print(age)

#             # cv2.putText(frame, f'{mssg}:{age}', (x,y - 10),
#             #             cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#             #             (0, 255, 255), 2, cv2.LINE_AA)
#             # cv2.rectangle(frame, (x, y), (x2, y2),
#             #             (00, 200, 200), 2)
#     # frame = video.read()
    
#     faces = face_cascade.detectMultiScale(gray,1.3,5)
#     for (x,y,w,h) in faces:  
#         face = frame[y-5:y+h+5,x-5:x+w+5]
#         resized_face = cv2.resize(face,(160,160))
#         resized_face = resized_face.astype("float") / 255.0
#         # resized_face = img_to_array(resized_face)
#         resized_face = np.expand_dims(resized_face, axis=0)
#         # pass the face ROI through the trained liveness detector
#         # model to determine if the face is "real" or "fake"
#         preds = modelAS.predict(resized_face)[0]
#         # print(preds)
#         if preds> 0.5:
#             label = 'spoof'
#             spoof_prediction.append(label)
#             cv2.putText(frame, f'{label}:{mssg}:{age}', (x,y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
#             cv2.rectangle(frame, (x, y), (x+w,y+h),
#                 (0, 0, 255), 2)
#         else:
#             label = 'real'
#             spoof_prediction.append(label)
#             cv2.putText(frame, f'{label}:{mssg}:{age}', (x,y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#             cv2.rectangle(frame, (x, y), (x+w,y+h),
#             (0, 255, 0), 2)
#     return frame


# class LivenessDetection(VideoTransformerBase):
#     def transform(self, frame):
#         frame = frame.to_ndarray(format="bgr24")
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         faces = face_detector(gray)
#         # If no faces our detected
#         if not faces:
#             msg = 'No face detected'
#             cv2.putText(frame, f'{msg}', (40, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 2, (200), 2)
#         else:
#             # --------- Bounding Face ---------#
#             for face in faces:
#                 x = face.left() # extracting the face coordinates
#                 y = face.top()
#                 x2 = face.right()
#                 y2 = face.bottom()
#                 # for box in Boxes:
#                 face = frame[y:y2, x:x2]
#                 # ----- Image preprocessing --------#
#                 blob = cv2.dnn.blobFromImage(
#                     face, 1.0, (227, 227), model_mean, swapRB=False)

#                 # -------Age Prediction---------#
#                 age_Net.setInput(blob)
#                 age_preds = age_Net.forward()
#                 # print(age_preds)
#                 age = ageList[age_preds[0].argmax()]
#                 # st.session_state.age_prediction.append(age)
#                 # age = '(8 - 12)'
#                 # print(age)

#                 # cv2.putText(frame, f'{mssg}:{age}', (x,y - 10),
#                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                 #             (0, 255, 255), 2, cv2.LINE_AA)
#                 # cv2.rectangle(frame, (x, y), (x2, y2),
#                 #             (00, 200, 200), 2)
#         # frame = video.read()
        
#         faces = face_cascade.detectMultiScale(gray,1.3,5)
#         for (x,y,w,h) in faces:  
#             face = frame[y-5:y+h+5,x-5:x+w+5]
#             resized_face = cv2.resize(face,(160,160))
#             resized_face = resized_face.astype("float") / 255.0
#             # resized_face = img_to_array(resized_face)
#             resized_face = np.expand_dims(resized_face, axis=0)
#             # pass the face ROI through the trained liveness detector
#             # model to determine if the face is "real" or "fake"
#             preds = modelAS.predict(resized_face)[0]
#             # print(preds)
#             if preds> 0.5:
#                 label = 'spoof'
#                 spoof_prediction.append(label)
#                 cv2.putText(frame, f'{label}:{mssg}:{age}', (x,y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
#                 cv2.rectangle(frame, (x, y), (x+w,y+h),
#                     (0, 0, 255), 2)
#             else:
#                 label = 'real'
#                 spoof_prediction.append(label)
#                 cv2.putText(frame, f'{label}:{mssg}:{age}', (x,y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#                 cv2.rectangle(frame, (x, y), (x+w,y+h),
#                 (0, 255, 0), 2)
#         return frame

 
#     # img = frame.to_ndarray(format="bgr24")

def main():
    # with st.sidebar:
    #     choose = option_menu("MEUNETS - Authentication", ["Authentication"],          
    #                         menu_icon="app-indicator", default_index=0,
    #                         styles={
    #         "container": {"padding": "5!important", "background-color": "#fafafa"},
    #         "icon": {"color": "orange", "font-size": "25px"}, 
    #         "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    #         "nav-link-selected": {"background-color": "#02ab21"},
    #     }
    #     )
    # Liveness Detection #
    # if choose == 'Authentication':
    st.title("Pain Management")
    # st.title("Real Time Liveness detection")
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
    # if st.button("Next"):
    #     print((spoof_prediction))
    # LivenessDetection.nikn()

    # elif choose == 'Age Detection':
    # st.title("Real Time Age detection")
    # RTC_CONFIGURATION = RTCConfiguration(
    #         {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    #     )
    # webrtc_ctx = webrtc_streamer(
    #         key="WYH",
    #         mode=WebRtcMode.SENDRECV,
    #         rtc_configuration=RTC_CONFIGURATION,
    #         media_stream_constraints={"video": True, "audio": False},
    #         video_processor_factory=AgeDetection,
    #         async_processing=True,
    #         )


    # elif choose == 'Kinship Verification':
    # st.title("Kinship Verification")
#     st.subheader("Please take two photos of yourself and your parent...")
#     img1 = st.file_uploader("Choose the photo of the parent")
#     img2 = st.file_uploader("Choose the photo of the child")
#     if img1 and img2 is not None:
#         img1_path = save_image(img1)
#         img2_path = save_image(img2)
#         res, pred = kinship_verify(img1_path, img2_path)
#         images = [img1_path, img2_path]
#         st.write (pred)
#         captions = ["Parent", "Childs\'"]
#         st.success("The model predicted as... " + str(res))
#         st.image(images, use_column_width=True, caption=captions)
#         # st.write ("The model predicted as...")
#         delete_img()
#     if st.button('Submit'):
#         # st.write('Age = ', st.session_state.age_prediction)
#         # choose  == 'Submit'
#         st.warning('Something went wrong! Please try again')
#         # st.success("You are authenticated successfully!!!")
#         # st.write("Redirect to the MEUNETS...")
#         # url = 'http://localhost:3000/my-profile/category'
#         # time.sleep(10)
#         # # if st.button('Next'):
#         # webbrowser.open_new_tab(url)
            
# # elif choose == "Submit":
# #      st.title("MEUNETS - Authentication")
# #      st.success("You are authenticated successfully!!!")
# #      st.write("Redirect to the MEUNETS...")
# #      url = 'http://localhost:3000/my-profile/category'

# #      if st.button('Next'):
# #         webbrowser.open_new_tab(url)

if __name__ == "__main__":
    main()



# video = cv2.VideoCapture(0)
# while True:
#     try:
#         ret,frame = video.read()
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray,1.3,5)
#         for (x,y,w,h) in faces:  
#             face = frame[y-5:y+h+5,x-5:x+w+5]
#             resized_face = cv2.resize(face,(160,160))
#             resized_face = resized_face.astype("float") / 255.0
#             # resized_face = img_to_array(resized_face)
#             resized_face = np.expand_dims(resized_face, axis=0)
#             # pass the face ROI through the trained liveness detector
#             # model to determine if the face is "real" or "fake"
#             preds = model.predict(resized_face)[0]
#             # print(preds)
#             if preds> 0.5:
#                 label = 'spoof'
#                 cv2.putText(frame, label, (x,y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
#                 cv2.rectangle(frame, (x, y), (x+w,y+h),
#                     (0, 0, 255), 2)
#             else:
#                 label = 'real'
#                 cv2.putText(frame, label, (x,y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#                 cv2.rectangle(frame, (x, y), (x+w,y+h),
#                 (0, 255, 0), 2)
#         cv2.imshow('frame', frame)
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#     except Exception as e:
#         pass
# video.release()        
# cv2.destroyAllWindows()