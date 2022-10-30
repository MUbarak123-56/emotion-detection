import streamlit as st
import pickle
import streamlit.components.v1 as components
import PIL
import io
import numpy as np
#import skimage
from pathlib import Path
#from fastai.vision.widgets import *
from fastai.vision.all import *
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(rc={'axes.facecolor':(0,0,0,0), 'figure.facecolor':(0,0,0,0)})
import time
from io import BytesIO
import base64


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
model = load_learner("model-pkl/resnet-50.pkl")

def run():
    st.set_page_config(layout='wide')
    header_image = Image.open("inside_out.png")
    #add_bg_from_local('inside_out_display.png')
    #st.sidebar.header("Emotion Detective")
    st.sidebar.markdown("<h1 style='text-align: center; color: white;'>Emotion Detective</h1>", unsafe_allow_html=True)
    st.subheader("Welcome to Emotion Detective! An emotion detection website")
    initial_sentence = "People display emotions multiple times when going through different events in life. They display negative emotions whenever something upsets them (anger, sadness, fear). They can be very happy when an event goes in their favour (happiness). They can be quite shocked at seeing the unexpected (surprised). And finally, they can display a neutral outlook when everything seems calm (neutral)."
    st.write(initial_sentence)
    objective_sentence = "The goal of this website will be to denote how someone feels by leveraging the power of Computer Vision to recognize and detect their emotional state. Some use cases of this app are interviewing, customer support and healthcare. Interviewers can utilize the website's ability to recognize interviewees' emotions and understand what their interviewees are going through during an interview. This will assist them with dealing with other interviewees in the future. Healthcare providers can also use this website's functionalities to know what a patient was feeling during a medical treatment, so they provide care for prospective patients without subjecting them to too much pain. Customer support representatives can use this website's ability to gain knowledge about how their customers feel so they can understand how to have better conversations with other customers in the future that will satisfy their customers' demands"
    st.write(objective_sentence)
   
    st.sidebar.image(header_image, use_column_width = True)
    st.write("Use the button below to upload an image of someone's face so we can detect its emotion. Possible emotional states are: angry, sad, happy, fearful, surprised and sad.")
    image_upload = st.file_uploader("Upload an image", type = ["png","jpg"])
    
    if image_upload is not None:
        st.balloons()
        img = Image.open(image_upload)
        img_display = img
        
        img = img.resize((48,48)).convert("L")
        img = PIL.Image.Image.to_bytes_format(img)
       
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
            
        tab1, tab2, tab3 = st.tabs(["Image","Emotion","Visualization"])
        
        tab1.image(img_display.resize((224,224)))
        
        image = Image.open("images/" + pred[0] + ".png")
        #image = image.resize((224,224))
        tab2.write("This person's emotional state is: " + pred[0])
        tab2.image(image)
        
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
        tab3.pyplot(fig)
        #buf = BytesIO()
        #fig.savefig(buf, format="png")
        #tab3.image(buf, width = 800, use_column_width = True)
    
if __name__ == '__main__':
    run()



