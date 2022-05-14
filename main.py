import streamlit as st
import streamlit.components.v1 as components
import cv2
import os

def cvtRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def resized(img):
    size = 512
    return cv2.resize(img,(size, size))
heading = f'<h1 style="font-weight:bold;text-align:center;">Topic</h1>'
st.markdown(heading, unsafe_allow_html=True)
# st.title('Betel vine Leaves Categorization and Disease Detection')
fs = st.file_uploader('upload Image',['jpg','png','jpeg'])
save_path = 'images/upload/'

isDir = os.path.isdir(save_path)
t=2
if not isDir :
    os.makedirs(save_path,0o666)

if fs is not None:
    img_path = os.path.join(save_path, fs.name)
    with open(img_path,'wb') as f:
        f.write(fs.read())


    img = cv2.imread(img_path)
    resized_img = resized(img)
    display_img = cvtRGB(resized_img)
    # display_img = re
    # image = Image.open(img_path)
    # image2 = image.rotate(-90)

    # st.image(image2)
    if fs:
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            st.write("")

        with col2:
            st.image(display_img)

        with col3:
            st.write("")

        input_img = os.path.join(save_path,fs.name)


        # result = healthy_classification.healthyLeafClassification(input_img)
        # if result == 0:
        #     new_title = '<p style="color:Green; font-size: 32px;font-weight:bold; text-align: center;">valid money</p>'
        #     st.markdown(new_title, unsafe_allow_html=True)
        #
        # else:
        #     new_title = '<p style="color:red; font-size: 32px; font-weight:bold; text-align: center;">Unhealthy Leaf</p>'
        #     st.markdown(new_title, unsafe_allow_html=True)
        #     print("unhealthy leaf")
