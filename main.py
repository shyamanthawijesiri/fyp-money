import streamlit as st
import streamlit.components.v1 as components
import cv2
import os
import predict_model as model

def cvtRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def resized(img):
    size = 512
    return cv2.resize(img,(size, size))
heading = f'<h1 style="font-weight:bold;text-align:center;">Topic</h1>'
st.markdown(heading, unsafe_allow_html=True)
fs = st.file_uploader('upload Image',['jpg','png','jpeg'])
save_path = 'images/upload/'

isDir = os.path.isdir(save_path)

if not isDir :
    os.makedirs(save_path,0o666)

if fs is not None:
    img_path = os.path.join(save_path, fs.name)
    with open(img_path,'wb') as f:
        f.write(fs.read())


    img = cv2.imread(img_path)
    resized_img = resized(img)
    display_img = cvtRGB(resized_img)

    if fs:
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            st.write("")

        with col2:
            st.image(display_img)

        with col3:
            st.write("")

        input_img = os.path.join(save_path,fs.name)
        money_img = cv2.imread(input_img)

        result = model.moneyClassification(money_img)
        if result == 1:
            new_title = '<p style="color:Green; font-size: 32px;font-weight:bold; text-align: center;">Valid Money</p>'
            st.markdown(new_title, unsafe_allow_html=True)

        else:
            new_title = '<p style="color:red; font-size: 32px; font-weight:bold; text-align: center;">Invalid Money</p>'
            st.markdown(new_title, unsafe_allow_html=True)

