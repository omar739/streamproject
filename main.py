import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import densenet201
from PIL import Image
@st.cache_resource
def load_the_model() :
     model = densenet201(pretrained=False)
     model.classifier = nn.Sequential(nn.Linear(1920, 512),nn.ReLU(),nn.Dropout(0.4),nn.Linear(512,7))
     model.load_state_dict(torch.load("D:/Courses/weights.pth"))
     model.eval()
     return model
 
def predicted_the_image(model,image) :
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),  
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])
    image = Image.open(image).convert("RGB")  
    image = transform(image).unsqueeze(0) 

    model.eval()  
    with torch.no_grad():
        output = model(image)  
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  
        predicted_class = torch.argmax(probabilities).item()  
    classes = ["Cas","Cos","Gum","MC","OC","OLP","OT"]    
    st.write(f"Predicted Class: {classes[predicted_class]}")
    st.bar_chart(probabilities.numpy()) 

st.title("Teeth Classification using AI")
st.subheader("model used is DenseNet-201 with total train accuracy 85 and validation accuracy 70, more epochs will increase the accuracy up to 95")
st.write("can be modified later because this is initial version of The web app")

st.sidebar.title("main page")
with st.sidebar :
    choose = st.selectbox("graphs or classifiy new image",["graphs describe training","classifiy and upload image"])

if choose == "graphs describe training" :
    tab1 , tab2 , tab3 , tab4 = st.tabs(["training graph","validation graph","classes distribution","model Evaluation"])
    
    with tab1 :
        train_acc_list = [
            0.3314, 0.5063, 0.5808, 0.6391, 0.6608, 0.6706, 0.6984, 0.7101, 0.7298, 0.7457,
            0.7360, 0.7454, 0.7454, 0.7645, 0.7577, 0.7655, 0.7684, 0.7862, 0.7891, 0.7917,
            0.7995, 0.8082, 0.8053, 0.8141, 0.8144, 0.8176, 0.8280, 0.8319, 0.8494, 0.8358,
            0.8377, 0.8348, 0.8280, 0.8384, 0.8426, 0.8452, 0.8455, 0.8597, 0.8555, 0.8610,
            0.8461, 0.8526, 0.8422, 0.8565, 0.8558, 0.8672, 0.8685, 0.8785, 0.8746, 0.8675
            ]
        train_loss_list = [
            1.7451, 1.3426, 1.1386, 1.0379, 0.9503, 0.8969, 0.8432, 0.8044, 0.7516, 0.7144,
            0.7149, 0.7018, 0.6949, 0.6607, 0.6534, 0.6414, 0.6294, 0.5885, 0.5781, 0.5666,
            0.5556, 0.5437, 0.5337, 0.5173, 0.5283, 0.5142, 0.4887, 0.4718, 0.4367, 0.4646,
            0.4442, 0.4492, 0.4765, 0.4378, 0.4420, 0.4344, 0.4135, 0.3880, 0.3948, 0.3967,
            0.4166, 0.3973, 0.4296, 0.3854, 0.3934, 0.3771, 0.3595, 0.3466, 0.3497, 0.3635
            ]
        epoches = [ i for i in range(1,51)]
        fig, ax = plt.subplots()
        plt.style.use("bmh")
        plt.plot(epoches,train_acc_list,color="black",linewidth=3)
        plt.xlabel("epoches")
        plt.ylabel("training accuracy")
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        plt.style.use("bmh")
        plt.plot(epoches,train_loss_list,color="blue",linewidth=3)
        plt.xlabel("epoches")
        plt.ylabel("training loss")
        st.pyplot(fig)
        st.write("as you see in this graph, increasing number of epoches will increase the accuracy")
        st.write("wait for the new update later with higher accuracy")
    with tab2 :
        val_acc_list = [
            0.3842, 0.4368, 0.4484, 0.4864, 0.4912, 0.5078, 0.5739, 0.5700, 0.5360, 0.5768,
            0.5895, 0.5730, 0.6099, 0.6031, 0.5739, 0.5710, 0.5525, 0.6342, 0.6099, 0.6099,
            0.6469, 0.6138, 0.6274, 0.6479, 0.6498, 0.6089, 0.6313, 0.6381, 0.6060, 0.6051,
            0.6187, 0.6352, 0.6080, 0.6362, 0.6440, 0.6430, 0.6566, 0.6488, 0.6381, 0.6216,
            0.6216, 0.6265, 0.6615, 0.6634, 0.6566, 0.6848, 0.6673, 0.6732, 0.6683, 0.6469
            ]
        val_loss_list = [
            1.6742, 1.5002, 1.4580, 1.3419, 1.2665, 1.2681, 1.1688, 1.1737, 1.2087, 1.1603,
            1.1428, 1.1362, 1.0438, 1.1855, 1.2440, 1.1748, 1.2503, 1.0204, 1.1456, 1.0974,
            0.9811, 0.9648, 1.0125, 0.9944, 0.9816, 1.0889, 1.0581, 1.0458, 1.0704, 1.0488,
            1.0608, 1.0674, 1.0675, 1.0005, 0.9832, 1.0019, 0.9485, 0.9542, 1.0028, 1.0604,
            1.0305, 0.9442, 0.9495, 0.9730, 1.0159, 0.9013, 0.9108, 0.8910, 0.9488, 1.0732
            ]
        
        epoches = [ i for i in range(1,51)]
        fig, ax = plt.subplots()
        plt.style.use("bmh")
        plt.plot(epoches,val_acc_list,color="green",linewidth=3)
        plt.xlabel("epoches")
        plt.ylabel("validation accuracy")
        st.pyplot(fig)
        fig, ax = plt.subplots()
        plt.style.use("bmh")
        plt.plot(epoches,val_loss_list,color="green",linewidth=3)
        plt.xlabel("epoches")
        plt.ylabel("validation loss")
        st.pyplot(fig)
        st.write("as you see in this graph, increasing number of epoches will increase the accuracy")
        st.write("wait for the new update later with higher accuracy")
        
    with tab3 :
        st.image("train data.png")
        st.subheader("this problem of biased classes can be solved with Weighted Cross Entropy by assign different weights inverse correlation with the rare of class")
        st.image("WCE.png")
        
    with tab4 :
        st.image("ccccc.png")
        st.header("Precision: 0.6785")
        st.header("Recall: 0.6423")
        
elif choose == "classifiy and upload image" :
    model = load_the_model()
    image = st.file_uploader(label="Upload image to classifiy here",type=["jpg","jpeg","png","bmp","webp"])
    if image is not None :
        st.image(image,caption="image you have uploaded")
        button = st.button(label="classifiy this image")
        
        if button :
            with st.spinner("wait few seconds please..") :
                predicted_the_image(model,image)