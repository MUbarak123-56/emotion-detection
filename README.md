# Emotion Detective
![python](https://img.shields.io/badge/Python-3.9.0%2B-blue)
[![View on Streamlit](https://img.shields.io/badge/Streamlit-View%20on%20Streamlit%20app-ff69b4?logo=streamlit)](https://emotion-detective.streamlit.app/)


# Directory
- [Project Overview](#project-overview)
- [Data](#data)
- [Modeling](#modeling)
- [Potential Use Cases](#potential-use-cases)
- [Resources](#resources)
- [Contributors](#contributors)

# Project Overview

This project mainly focuses on utilising the technology of deep learning models to detect different emotions of people in an image. Emotion detection is a way to understand people better in social settings to detect feelings like happiness, sadness, surprise at a specific moment without actually asking them. It is useful in many areas like security, investigation and healthcare. After the algorithm was built, an emotion detection website was developed to allow users to upload images, and get the appropriate emotion of the person in the image.

# Data

Data source: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

The data was downloaded from Kaggle, and it contains 35,685 emotion images in total and categorised them into 7 different categories (sad, neutral, happy, angry, disgusted, surprised, fearful). All the emotion images are saved in png format and each of them has a shape of 48x48 pixels in grayscale. However, we realised that the data labels were not all sufficiently represented: all the other emotion categories have over 4,000 image data, while for the disgusted category had about 500 images available. This can potentially cause problems later in the project. Therefore, we decided to remove the disgusted category from the data, which means only six emotions (sad, neutral, happy, angry, surprised, fearful) will be used for classification. This is done to avoid a data imbalance problem. 

# Modeling
## Modeling Approaches
AlexNet vs VGG vs ResNet (TODO)

## Model Training
(TODO)

# Potential Use Cases 

Although the app's functionality is based on static images. In the long run, we would love to develop a model/app that can work dynamically (i.e. take motion into account to record someone's emotion). This would mean that someone's emotional state will be recorded over the course of a certain period of time when looking into a camera. Then, the emotional states will be used to know how the person felt at different points of the conversation. They can also build emotional frequency charts to display how the frequency of the person's emotional state over the course of the conversation. Some use cases of this app are interviewing, customer support and healthcare. Interviewers can utilize the website's ability to recognize interviewees' emotions and understand what their interviewees are going through during an interview. This will assist them with dealing with other interviewees in the future. Healthcare providers can also use this website's functionalities to know what a patient was feeling during a medical treatment, so they provide care for prospective patients without subjecting them to too much pain. Customer support representatives can use this website's ability to gain knowledge about how their customers feel so they can understand how to have better conversations with other customers in the future that will satisfy their customers' demands

# Resources
- FastAI
- Streamlit
- Google Colab
- PyTorch
- MTCNN
- OpenCV
- Seaborn
- NumPy
- Matplotlib
# Contributors
- Mubarak Ganiyu (mubarak.a.ganiyu@vanderbilt.edu)
- Tinglei Wu (tinglei.wu@vanderbilt.edu)
- Alvin Chen (yiwen.chen@vanderbilt.edu)
