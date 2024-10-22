import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from PIL import Image
import streamlit as st
import google.generativeai as genai

# Streamlit Page Configuration
st.set_page_config(layout="wide")
st.title("GestureLens")
st.image('AI.png')

# Columns for layout
col1, col2 = st.columns([3, 2])

with col1:
    run_button = st.button('Run')
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

# Configure Google GenAI
genai.configure(api_key="AIzaSyAu7w2tMO4kIAiB-RDMh8vywmF8OqBjpQk")
model = genai.GenerativeModel('gemini-1.5-flash')

# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=0, detectionCon=0.7, minTrackCon=0.5)

# Function to get hand information
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

# Function to draw on the canvas
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, (0, 0, 255), 15)  # Red line
    elif fingers == [0, 1, 1, 1, 1]:
        canvas = np.zeros_like(canvas)  # Clear canvas gesture

    return current_pos, canvas

# Function to send image to AI
def sendToAI(model, canvas, fingers):
    if fingers == [1, 0, 0, 0, 0]:  # Gesture for AI request
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["guess the image", pil_image])
        return response.text

# Initial variables
prev_pos = None
canvas = None
output_text = ""

# Running the main loop on button press
if run_button:
    success, img = cap.read()

    if not success:
        st.warning("Webcam not found or not accessible.")
    else:
        img = cv2.flip(img, 1)

        if canvas is None:
            canvas = np.zeros_like(img)

        info = getHandInfo(img)
        if info:
            fingers, lmList = info
            prev_pos, canvas = draw(info, prev_pos, canvas)
            output_text = sendToAI(model, canvas, fingers)

        # Combine webcam image and drawing
        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        FRAME_WINDOW.image(image_combined, channels="BGR")

        if output_text:
            output_text_area.text(output_text)

cap.release()
cv2.destroyAllWindows()
