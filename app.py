#import libraries
#import cv2
from mtcnn.mtcnn import MTCNN
import cv2
from matplotlib import pyplot
import streamlit as st
import streamlit.components.v1 as components
import PIL
import io
import numpy as np
#from pathlib import Path
from fastai.vision.all import *
#import pathlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(rc={'axes.facecolor':(0,0,0,0), 'figure.facecolor':(0,0,0,0)})
import time
from io import BytesIO
import base64

#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath
model = load_learner("model-pkl/resnet-50.pkl")
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_classifier = cv2.CascadeClassifier (cv2.data.haarcascades + 'haarcascade_eye.xml')
eye_classifier = cv2.CascadeClassifier (cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
mouth_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
nose_classifier = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

detector = MTCNN()

def face_detector(img):
    #gray = img
    # Convert Image to Grayscale
    #if img.shape == 3:
    #    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #elif img.shape == 2:
    #    gray = img
        
    if (img.shape[0] == 48) & (img.shape[1] == 48) & (img.shape == 2):
        new_img = PIL.Image.fromarray(img)
        #new_img = new_img.resize((48,48)).convert("L")
        return new_img
        
    elif img.shape == 3:
        faces = detector.detect_faces(img)
    
        if (len(faces) != 1):
            return False
        elif len(faces) == 1:
        #x1, y1, width, height = faces[0]['box']
        #x2, y2 = x1 + width, y1 + height
        #cropped_face = img[y1:y2, x1:x2]
        #new_img = PIL.Image.fromarray(cropped_face)
        #new_img = new_img.resize((48,48)).convert("L")
        #return new_img
            gray = img
        # Convert Image to Grayscale
            #if img.shape == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #elif img.shape == 2:
            #    gray = img
        
            if (gray.shape[0] < 112) | (gray.shape[1] < 112):
                arr_img = PIL.Image.fromarray(gray)
                width, height = arr_img.size
                ratio = width/height
                new_height = 224
                new_width = ratio*new_height
                arr_img = arr_img.resize((int(new_width),int(new_height)))
                arr_img = np.array(arr_img)
                if arr_img.shape == 3:
                    gray = cv2.cvtColor(arr_img, cv2.COLOR_BGR2GRAY)
                elif arr_img.shape == 2:
                    gray = arr_img
        
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=10, minSize=(30, 30))
            eyes = eye_classifier.detectMultiScale(gray)
            mouth = mouth_classifier.detectMultiScale(gray)
            nose = nose_classifier.detectMultiScale(gray)
    
            if (len(faces) != 1) & (len(eyes) < 1) & (len(mouth) < 1) & (len(nose) < 1):
                return False
            elif len(faces) == 1:
                x, y, w, h = faces[0]
                cropped_img = cv2.rectangle(img[y:y+h, x:x+w], (0, 0), (0,0), (0, 0, 0), 2)
                new_img = PIL.Image.fromarray(cropped_img)
                new_img = new_img.resize((48,48)).convert("L")
                return new_img
            elif len(eyes) > 0:
                new_img = PIL.Image.fromarray(img)
                new_img = new_img.resize((48,48)).convert("L")
                return new_img
            elif (len(mouth) > 0):
                new_img = PIL.Image.fromarray(img)
                new_img = new_img.resize((48,48)).convert("L")
                return new_img
            elif (len(nose) > 0):
                new_img = PIL.Image.fromarray(img)
                new_img = new_img.resize((48,48)).convert("L")
                return new_img
    
def run():
    st.set_page_config(layout='wide')
    header_image = Image.open("inside_out.png")
    st.markdown("<h1 style='text-align: center; color: white;'>Emotion Detective</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.write(' ')
    col2.image(header_image, caption = "Photo Credit: https://techcrunch.com/2015/04/12/how-pixar-solves-problems-from-the-inside-out/", use_column_width = True)
    col2.write(' ')
    st.markdown("<h2 style='text-align: center; color: white;'>Welcome to Emotion Detective! An emotion detection website</h2>", unsafe_allow_html=True)
    initial_sentence = "People display emotions multiple times when going through different events in life. They display negative emotions whenever something upsets them (anger, sadness, fear). They can be very happy when an event goes in their favour (happiness). They can be quite shocked at seeing the unexpected (surprised). And finally, they can display a neutral outlook when everything seems calm (neutral)."
    st.write(initial_sentence)
    objective_sentence = "The goal of this website will be to denote how someone feels by leveraging the power of Computer Vision to recognize and detect their emotional state. Some use cases of this app are interviewing, customer support and healthcare. Interviewers can utilize the website's ability to recognize interviewees' emotions and understand what their interviewees are going through during an interview. This will assist them with dealing with other interviewees in the future. Healthcare providers can also use this website's functionalities to know what a patient was feeling during a medical treatment, so they provide care for prospective patients without subjecting them to too much pain. Customer support representatives can use this website's ability to gain knowledge about how their customers feel so they can understand how to have better conversations with other customers in the future that will satisfy their customers' demands"
    st.write(objective_sentence)
    st.write("Use the button below to upload an image of someone's face (preferably a headshot) so we can detect its emotion. Possible emotional states are: angry, happy, fearful, neutral, sad and surprised.")
    st.markdown("<h7 style='text-align: center; color: red;'>Note 1: Please, use your computer to upload an image. This is better than taking a selfie from your phone or using an image from your phone. This approach tend to yield better results. What you can do is take some pictures and transfer them to your computer so you can upload them to this website.</h7>", unsafe_allow_html=True)
    st.markdown("<h7 style='text-align: center; color: red;'>Note 2: If you are using a phone to take a selfie to upload an image for analysis, you will have to set your phone horizontal (i.e. rotate your phone by 90 degrees to the left or 90 degrees to the right) while taking the picture for it to be properly processed for analysis</h7>", unsafe_allow_html=True)
    st.markdown("<h7 style='text-align: center; color: red;'>Note 3: The model might be terrible at detecting the toddler/children's faces, so it will yield an error upon uploading their images.</h7>", unsafe_allow_html=True)
    image_upload = st.file_uploader("Upload an image", type = ["png","jpg","jpeg","jfif"])
    
    if image_upload is not None:
        image_open = Image.open(image_upload)
        img_array = np.array(image_open)
        check_img = face_detector(img_array)
        if check_img is False:
            st.error("There is an error with your image")
            st.markdown("<h7 style='text-align: center; color: white;'>You can fix this by following one or all of the instructions below:</h7>", unsafe_allow_html=True)
            st.markdown("<h9 style='text-align: center; color: white;'>- Upload an image with a face looking inside the camera.</h9>", unsafe_allow_html=True)
            st.markdown("<h9 style='text-align: center; color: white;'>- Upload an image with only one person's face.</h9>", unsafe_allow_html=True)
            st.markdown("<h9 style='text-align: center; color: white;'>- Upload an image with a face that is very visible</h9>", unsafe_allow_html=True)
        else: 
            img = PIL.Image.Image.to_bytes_format(check_img)
            st.balloons()
            img_display = Image.open(image_upload)
        
            ## Inferencing image
            pred = model.predict(img)
            my_bar = st.progress(0)
            st.write("Generating Results")
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1)
            #st.write(pred)
            pred_labels = ["angry","fearful","happy","neutral", "sad", "surprised"]
            pred_values = pred[2].tolist()
            for i in range(len(pred_values)):
                pred_values[i] = round(pred_values[i],3)
            
            tab1, tab2, tab3, tab4 = st.tabs(["Image","Face","Emotion","Visualization"])
        
            width, height = img_display.size
            ratio = width/height
            height = 224
            width = ratio*224
            tab1.image(img_display.resize((int(width),int(height))))
        
            c_width, c_height = check_img.size
            c_ratio = c_width/c_height
            c_height = 224
            c_width = c_ratio*224
            tab2.image(check_img.resize((int(c_width),int(c_height))))
        
            image = Image.open("images/" + pred[0] + ".png")
            #image = image.resize((224,224))
            tab3.write("This person's emotional state is: " + pred[0])
            tab3.image(image)
        
            df_plot = pd.DataFrame({"label":pred_labels,"value":pred_values})
            df_plot = df_plot.sort_values("value", ascending = False)
        
            fig, ax = plt.subplots(figsize=(12,8))
            sns.barplot(x="value", y="label", data=df_plot, color = "blue")
            ax.set_title("Emotional Rankings",fontdict= {'fontsize': 20, 'fontweight':'bold'})
            ax.set_xlabel("Probability")
            ax.set_ylabel("Label")
            ax.xaxis.label.set_color('white')       
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.tick_params(axis='x', colors='white')   
            ax.tick_params(axis='y', colors='white')
            tab4.pyplot(fig)
        
if __name__ == '__main__':
    run()



