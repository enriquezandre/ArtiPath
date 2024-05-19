from ultralytics import YOLO
import time
import streamlit as st
import cv2
import os
import tempfile
import settings


def load_model(model_path):
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def play_webcam(conf, model):
    # Add a sidebar option to select the camera source
    camera_source = st.sidebar.selectbox('Select Camera Source', (0, 1, 2, 3, 4))

    source_webcam = cv2.VideoCapture(camera_source)
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = source_webcam
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_video(conf, model):
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=['mp4', 'mov', 'avi'])

    is_display_tracker, tracker = display_tracker_options()

    temp_file_path = None
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Display the video
        st.video(uploaded_file)
    else:
        st.video("videos/testvid_1.mp4")

    if st.sidebar.button('Detect Video Objects'):
        try:
            # Use the temporary file path if uploaded file is present, else use default video
            vid_path = temp_file_path if temp_file_path else "videos/testvid_1.mp4"
            vid_cap = cv2.VideoCapture(vid_path)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
        finally:
            # Delete the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)