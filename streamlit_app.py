from pprint import pprint
import streamlit as st
import requests
import base64
import io
from PIL import Image
import torch
import numpy as np
import cv2
from matplotlib import cm
import pandas as pd


#Import the model and the weights
model = torch.hub.load(r"D:\Git_Repo\Trash_Categorizing\yolov5-master", 'custom', path=r"D:\Git_Repo\Trash_Categorizing\yolov5 training weights\second.pt", source = 'local', force_reload=True)

# Add in location to select image.
st.sidebar.write('#### Select the app mode you want to use.')

#creating the mode choice button
mode = ["About", "Photo", "Camera detection"]
choice = st.sidebar.selectbox("Select Activity", mode)

if choice == "Photo":
    st.sidebar.write('#### Select an image to upload.')

    uploaded_file = st.sidebar.file_uploader('',
                                            type=['png', 'jpg', 'jpeg'],
                                            accept_multiple_files=False)

    ## Title.
    st.write('# Trash type Detection')

    ## Pull in default image or user-selected image.
    if uploaded_file is None:
        # Default image.
        url = 'https://raw.githubusercontent.com/pyramixofficial/My-personal-projects/main/ML%20-%20Trash%20Categorizing/images/trash_img.jpg'
        image = Image.open(requests.get(url, stream=True).raw)

    else:
        # User-selected image.
        image = Image.open(uploaded_file)

    ## Subtitle.
    st.write('### Inferenced Image')

    #Processing and saving the image 
    image_result = model(image, size = 640)
    image_result.save(save_dir = r'D:\Git_Repo\Trash_Categorizing\results')

    #Opening the saved image
    result_image = Image.open(r'D:\Git_Repo\Trash_Categorizing\results\image0.jpg')
    
    # Display image.
    st.image(result_image,
            caption= 'Processed image',
            use_column_width=True)

    st.write("### Save the processed data to a csv here:)")

    # Saving the data into a csv
    if st.button("Save data"):
        t_np = image_result.__dict__['pred'][0].cpu().numpy() #convert to Numpy array
        df = pd.DataFrame(t_np, columns=['x_first', 'y_first', 'x_second', 'y_second', 'probability', 'type']) #convert to a dataframe
        df['type'] = df['type'].apply(lambda x: image_result.__dict__['names'][int(x)])
        df.to_csv("data.csv") #save to file
        st.write("Data was saved to data.csv")


#Settin the camera detection option
elif choice == "Camera detection":
    st.header("Live Camera")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    while run:
        #Reading and converting the image
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Making predictions on the image and printing it on the app
        frame = model(frame, size=640)
        FRAME_WINDOW.image(frame.render())
        
    else:
        st.write('Stopped')

#The description page of the project
elif choice == "About":
    st.write("###### This is a project created by Eduard Balamatiuc")
    st.write("---")
    st.write("### The problem addressed:")
    st.write("Serious pollution of the environment, which leads to damage to human health and the quality of ecosystems is a major problem today. Large unmonitored areas and overcrowded with waste are red areas that indirectly end up causing diseases that sometimes lead to deaths.")
    st.write("---")
    st.write("### Our solution to the problem")
    st.write("> Our project represents a tool for trash detection, that can identify the trash types and give you an output with coordinates on what the model has found in the input. Besides that you can get a livetime feed of the identified objects that will be placed inside a square.")
    st.write("#### There are two possible settings:")
    st.write("- Image")
    st.write("The Image option works based on already saved images. You need to browse and select an image from your device and afterwards the model will return you the image with identified trash objects.")
    st.image("git_images/image_option.png")
    st.write("Besides that, the user has an option of exporting the data in a csv file, where he can get: the number of identified objects, the coordinated for the two points that form the square in which the identified object is placed, the probability of the detection and the type of the identified object.")
    st.image("git_images/data_saving.png")
    st.write("---")
    st.write("- Video")
    st.write("  This part of the project is extremely simple and useful. First of all make sure that you have a camera on your device, afterwards, check the box Run to start the project and allow your camera to be used. Then a window with the live dstribution of the camera will appear below and you will be able to see the identified results instantly.")
    st.image("git_images/camera_option.png")
    st.write("> An important detail to mention is that on the Video trash detection option the model is not always extremely accurate, but it still manages to make decent predictions.")
    st.image("git_images/ezgif.com-gif-maker.gif")











