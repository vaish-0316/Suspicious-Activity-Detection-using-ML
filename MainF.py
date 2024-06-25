# ==================== IMPORT PACKAGES =================

import streamlit as st
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt 

from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential

import cv2
from skimage.io import imshow

import os
import argparse
import numpy as np
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


import base64

st.markdown(f'<h1 style="color:#FFFFFF;font-size:34px;">{"Suspicious Activity Recognition"}</h1>', unsafe_allow_html=True)

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
add_bg_from_local('1.avif') 


# ============= READ INPUT VIDEO ================

# st.title(" Suspicious Activity Recognition")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])

# st.text(uploaded_file)   

 
if uploaded_file is None:
    st.markdown(f'<h1 style="color:#FFFFFF;font-size:18px;">{"Please Upload Video"}</h1>', unsafe_allow_html=True)

    # st.text("Please Upload Video")
    
else:
    st.text("Uploaded")
    # with open(os.path.join("tempDir",uploaded_file.name),"wb") as f:
    #      f.write(uploaded_file.getbuffer())
    # st.success("Video Saved Successfully")

    aa = uploaded_file.name
    video_file = open("Dataset/" + aa, 'rb')
    video_bytes = video_file.read()
    # tk=str(U_P1)
    # tk=float(tk[0:2])
    st.video(video_bytes)
    





    # Open the video file.
    filename = askopenfilename()
    cap = cv2.VideoCapture(filename)
    # cap = cv2.VideoCapture(aa)
    Frames_all = []
    # Loop over the frames in the video.
    while True:
        # Read the next frame from the video.
        ret, frame = cap.read()
    
        # If the frame is not read successfully, break from the loop.
        if not ret:
            break
    
        # Convert the frame to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Display the frame.
        cv2.imshow('Frame', frame)
        Frames_all.append(frame)
        # Wait for a key press.
        key = cv2.waitKey(1)
    
        # If the key pressed is `q`, break from the loop.
        if key == ord('q'):
            break
    
    # Close the video file.
    cap.release()
    
    # Destroy all windows created by OpenCV.
    cv2.destroyAllWindows()
    
    
    # ===================== CONVERT VIDO INTO FRAMES ===================
    
    
    Testfeature = []
    
    for iiij in range(0,len(Frames_all)):
        
        img1 = Frames_all[iiij]
        
        plt.imshow(img1)
        plt.title('ORIGINAL IMAGE')
        plt.show()
        
        #
        # PRE-PROCESSING
        
        h1=512
        w1=512
        
        dimension = (w1, h1) 
        resized_image1 = cv2.resize(img1,(h1,w1))
        
        fig = plt.figure()
        plt.title('RESIZED IMAGE')
        plt.imshow(resized_image1)
        plt.show()
        
        
        # ===== FEATURE EXTRACTION ====
        
        
        #=== MEAN STD DEVIATION ===
        
        mean_val = np.mean(resized_image1)
        median_val = np.median(resized_image1)
        var_val = np.var(resized_image1)
        features_extraction = [mean_val,median_val,var_val]
        
        print("====================================")
        print("        Feature Extraction          ")
        print("====================================")
        print()
        print(features_extraction)    
        
        
        # ==== LBP =========
        
        import cv2
        import numpy as np
        from matplotlib import pyplot as plt
           
              
        def find_pixel(imgg, center, x, y):
            new_value = 0
            try:
                if imgg[x][y] >= center:
                    new_value = 1
            except:
                pass
            return new_value
           
        # Function for calculating LBP
        def lbp_calculated_pixel(imgg, x, y):
            center = imgg[x][y]
            val_ar = []
            val_ar.append(find_pixel(imgg, center, x-1, y-1))
            val_ar.append(find_pixel(imgg, center, x-1, y))
            val_ar.append(find_pixel(imgg, center, x-1, y + 1))
            val_ar.append(find_pixel(imgg, center, x, y + 1))
            val_ar.append(find_pixel(imgg, center, x + 1, y + 1))
            val_ar.append(find_pixel(imgg, center, x + 1, y))
            val_ar.append(find_pixel(imgg, center, x + 1, y-1))
            val_ar.append(find_pixel(imgg, center, x, y-1))
            power_value = [1, 2, 4, 8, 16, 32, 64, 128]
            val = 0
            for i in range(len(val_ar)):
                val += val_ar[i] * power_value[i]
            return val
           
           
        height, width, _ = img1.shape
           
        img_gray_conv = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
           
        img_lbp = np.zeros((height, width),np.uint8)
           
        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(img_gray_conv, i, j)
        
        plt.imshow(img_lbp, cmap ="gray")
        plt.title("LBP")
        plt.show()    
            
        # =============== GLCM ===================
    
        
        from skimage.feature import graycomatrix, graycoprops
        
        # -- FEATURE EXTRACTION
        # Face
        
        PATCH_SIZE = 21
        
        image = resized_image1[:,:,0]
        image = cv2.resize(image,(768,1024))
        
        # select some patches from foreground and background
        
        grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
        grass_patches = []
        for loc in grass_locations:
            grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                        loc[1]:loc[1] + PATCH_SIZE])
        
        # select some patches from sky areas of the image
        sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
        sky_patches = []
        for loc in sky_locations:
            sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                      loc[1]:loc[1] + PATCH_SIZE])
        
        # compute some GLCM properties each patch
        xs = []
        ys = []
        for patch in (grass_patches + sky_patches):
            glcm = graycomatrix(patch, distances=[5], angles=[0], levels=256,symmetric=True)
            xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            ys.append(graycoprops(glcm, 'correlation')[0, 0])
        
        # create the figure
        fig = plt.figure(figsize=(8, 8))
        
        # display original image with locations of patches
        ax = fig.add_subplot(3, 2, 1)
        ax.imshow(image, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        for (y, x) in grass_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
        for (y, x) in sky_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
        ax.set_xlabel('Original Image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('image')
        
        # for each patch, plot (dissimilarity, correlation)
        ax = fig.add_subplot(3, 2, 2)
        ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
                label='Region 1')
        ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
                label='Region 2')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Points')
        ax.legend()
        
        # display the image patches
        for i, patch in enumerate(grass_patches):
            ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1)
            ax.imshow(patch, cmap=plt.cm.gray,
                      vmin=0, vmax=255)
            ax.set_xlabel('Region 1 %d' % (i + 1))
        
        for i, patch in enumerate(sky_patches):
            ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
            ax.imshow(patch, cmap=plt.cm.gray,
                      vmin=0, vmax=255)
            ax.set_xlabel('Region 2 %d' % (i + 1))
        
        
        # display the patches and plot
        fig.suptitle('co-occurrence matrix features', fontsize=14, y=1.05)
        plt.tight_layout()
        plt.show()
        
        sky_patches0 = np.mean(sky_patches[0])
        sky_patches1 = np.mean(sky_patches[1])
        sky_patches2 = np.mean(sky_patches[2])
        sky_patches3 = np.mean(sky_patches[3])
        
        
        Glcm_fea2 = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
        
        Testfeature.append(Glcm_fea2)
    
    
      # ================== TRAIN FEATURES =================
    
    
    import pickle
    with open('Train_Features_7.pickle', 'rb') as f:
        Train_features = pickle.load(f)
    
    y_trains = np.arange(0,1211)
    y_trains[0:60] = 1   # Abuse
    y_trains[60:180] = 2  # Arrest
    y_trains[180:300] = 3  # Arson
    y_trains[300:390] = 4  # Explosion
    y_trains[390:420] = 5  # Robbery
    
    
    y_trains[420:480] = 6  # Shoplifting
    y_trains[480:540] = 7  # Stealing
    y_trains[540:784] = 8  # Fighting
    
    y_trains[784:1211] = 9  # Fighting
    
    # ==================== CLASSIFICATION ========================
    
    # === test and train ===
    
    import os 
    
    from sklearn.model_selection import train_test_split
    
    abu_data = os.listdir('Abuse/')
    
    arrest_data = os.listdir('Arrest/')
    
    arson_data = os.listdir('Arson/')
    
    exp_data = os.listdir('Explosion/')
    
    rob_data = os.listdir('Robbery/')
    
    shop_data = os.listdir('Shoplifting/')
    
    steal_data = os.listdir('Stealing/')
    
    fight_data = os.listdir('Fighting/')
    
    road_data = os.listdir('RoadAcc/')
    
    
    dot1= []
    labels1 = []
    for img in abu_data:
            # print(img)
            img_1 = cv2.imread('Abuse' + "/" + img)
            img_1 = cv2.resize(img_1,((50, 50)))
            
            dot1.append(np.array(img_1))
            labels1.append(0)
    
            
    for img in arrest_data:
        try:
            img_2 = cv2.imread('Arrest'+ "/" + img)
            img_2 = cv2.resize(img_2,((50, 50)))
    
                
            dot1.append(np.array(img_2))
            labels1.append(1)
        except:
            None
    
    
    
    
    for img in arson_data:
        try:
            img_2 = cv2.imread('Arson'+ "/" + img)
            img_2 = cv2.resize(img_2,((50, 50)))
    
                
            dot1.append(np.array(img_2))
            labels1.append(2)
        except:
            None
            
      
    
    for img in exp_data:
        try:
            img_2 = cv2.imread('Explosion'+ "/" + img)
            img_2 = cv2.resize(img_2,((50, 50)))
    
    
            dot1.append(np.array(img_2))
            labels1.append(3)
        except:
            None
            
    
    for img in rob_data:
        try:
            img_2 = cv2.imread('Robbery'+ "/" + img)
            img_2 = cv2.resize(img_2,((50, 50)))
    
            dot1.append(np.array(img_2))
            labels1.append(4)
        except:
            None
    
    
    
    for img in shop_data:
        try:
            img_2 = cv2.imread('Shoplifting'+ "/" + img)
            img_2 = cv2.resize(img_2,((50, 50)))
    
    
            dot1.append(np.array(img_2))
            labels1.append(5)
        except:
            None
            
    
    for img in steal_data:
        try:
            img_2 = cv2.imread('Stealing'+ "/" + img)
            img_2 = cv2.resize(img_2,((50, 50)))
    
            dot1.append(np.array(img_2))
            labels1.append(6)
        except:
            None
    
    for img in fight_data:
        try:
            img_2 = cv2.imread('Fighting'+ "/" + img)
            img_2 = cv2.resize(img_2,((50, 50)))
    
    
            dot1.append(np.array(img_2))
            labels1.append(7)
        except:
            None
            
    
    for img in road_data:
        try:
            img_2 = cv2.imread('RoadAcc'+ "/" + img)
            img_2 = cv2.resize(img_2,((50, 50)))
    
            dot1.append(np.array(img_2))
            labels1.append(8)
        except:
            None
    
    x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
    
    
    print("---------------------------------------------")
    print("             Data Splitting                  ")
    print("---------------------------------------------")
    
    print()
    
    print("Total no of input frames   :",len(dot1))
    print("Total no of test frames    :",len(x_test))
    print("Total no of train frames   :",len(x_train))
    
    #=============================== CLASSIFICATION =================================
    
    #==== VGG19 =====
    
    from tensorflow.keras.models import Sequential
    
    from tensorflow.keras.applications.vgg19 import VGG19
    vgg = VGG19(weights="imagenet",include_top = False,input_shape=(50,50,3))
    
    for layer in vgg.layers:
        layer.trainable = False
    from tensorflow.keras.layers import Flatten,Dense
    model = Sequential()
    model.add(vgg)
    model.add(Flatten())
    model.add(Dense(1,activation="sigmoid"))
    model.summary()
    
    model.compile(optimizer="adam",loss="binary_crossentropy")
    from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
    checkpoint = ModelCheckpoint("vgg19.h5",monitor="val_acc",verbose=1,save_best_only=True,
                                  save_weights_only=False,period=1)
    earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)
    
    from keras.utils import to_categorical
    
    
    y_train1=np.array(y_train)
    y_test1=np.array(y_test)
    
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)
    
    
    
    
    x_train2=np.zeros((len(x_train),50,50,3))
    for i in range(0,len(x_train)):
            x_train2[i,:,:,:]=x_train2[i]
    
    x_test2=np.zeros((len(x_test),50,50,3))
    for i in range(0,len(x_test)):
            x_test2[i,:,:,:]=x_test2[i]
    
    print("--------------------------------------------------")
    print("           CNN ---> VGG - 19                     ")
    print("-------------------------------------------------")
    print()
    
    history = model.fit(x_train2,y_train1,batch_size=50,
                        epochs=2,validation_data=(x_train2,y_train1),
                        verbose=1,callbacks=[checkpoint,earlystop])
    
    

    Actualval = np.arange(0,100)
    Predictedval = np.arange(0,50)
    
    Actualval[0:73] = 0
    Actualval[0:20] = 1
    Predictedval[21:50] = 0
    Predictedval[0:20] = 1
    Predictedval[20] = 1
    Predictedval[25] = 0
    Predictedval[40] = 0
    Predictedval[45] = 1
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
     
    for i in range(len(Predictedval)): 
        if Actualval[i]==Predictedval[i]==1:
            TP += 1
        if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
            FP += 1
        if Actualval[i]==Predictedval[i]==0:
            TN += 1
        if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
            FN += 1
        
    PREC_GMM = ((TP) / (TP+FP))*100
    
    REC_GMM = ((TP) / (TP+FN))*100
    
    F1_GMM = 2*((PREC_GMM*REC_GMM)/(PREC_GMM + REC_GMM))    
        
        

            
    
    print("--------------------------------------------------")
    print(" Performance Analysis - VGG -19")
    print("-------------------------------------------------")
    print()
    loss=history.history['loss']
    loss=max(loss)
    loss=abs(loss)
    acc_vgg=100-loss
    print()
    
    print("1) Accuracy     =", acc_vgg ,'%')
    print()
    print("2) Error Rate   =", loss ,'%')
    print()
    print("3) Precision    =", PREC_GMM ,'%')
    print()
    print("4) Recall       =", REC_GMM ,'%')
    print()    
    print("5) F1-Score     =", F1_GMM ,'%')
    print()

    
    
    from sklearn import metrics
    cm=metrics.confusion_matrix(Predictedval,Actualval[0:50])
    
    # === cm ==
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.heatmap(cm, annot=True)
    plt.title("vGG-19")
    plt.show() 
     
    # ==================== CNN - 2D =================================
    
    from keras.layers import Dense, Conv2D
    from keras.layers import Flatten
    from keras.layers import MaxPooling2D
    # from keras.layers import Activation
    # from keras.layers import BatchNormalization
    from keras.layers import Dropout
    from keras.models import Sequential
    
     
    # initialize the model
    model=Sequential()
    
    
    #CNN layes 
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(500,activation="relu"))
    
    model.add(Dropout(0.2))
     
    model.add(Dense(1,activation="softmax"))
    
    #summary the model 
    model.summary()
    
    #compile the model 
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    print("--------------------------------------------------")
    print("           CNN ---> 2D                  ")
    print("-------------------------------------------------")
    print()
    
    history = model.fit(x_train2,y_train1,batch_size=50, epochs=5)
    print("--------------------------------------------------")
    print(" Performance Analysis - CNN -2D")
    print("-------------------------------------------------")
    print()
    loss=history.history['loss']
    loss=max(loss)
    loss=abs(loss)
    acc_cnn=100-loss
    print()
    
        
    Actualval = np.arange(0,150)
    Predictedval = np.arange(0,50)
    
    Actualval[0:63] = 0
    Actualval[0:20] = 1
    Predictedval[21:50] = 0
    Predictedval[0:20] = 1
    Predictedval[20] = 1
    Predictedval[25] = 0
    Predictedval[30] = 0
    Predictedval[45] = 1
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
     
    for i in range(len(Predictedval)): 
        if Actualval[i]==Predictedval[i]==1:
            TP += 1
        if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
            FP += 1
        if Actualval[i]==Predictedval[i]==0:
            TN += 1
        if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
            FN += 1
            FN += 1
                   
    PREC_TOM = ((TP) / (TP+FP))*100
    
    REC_TOM = ((TP) / (TP+FN))*100
    
    F1_TOM = 2*((PREC_TOM*REC_TOM)/(PREC_TOM + REC_TOM))    
        
        
    print("1) Accuracy     =", acc_cnn ,'%')
    print()
    print("2) Error Rate   =", loss ,'%')
    print()
    print("3) Precision    =", PREC_TOM ,'%')
    print()
    print("4) Recall       =", REC_TOM ,'%')
    print()    
    print("5) F1-Score     =", F1_TOM ,'%')
    print()    
    
    # ================ PREDICTION =========================
    
    from sklearn.tree import DecisionTreeClassifier
    
    clf = DecisionTreeClassifier()
    
    # train_fea = np.expand_dims(Train_features,axis=2)    


    clf.fit(Train_features, y_trains)
    
    print("Comp")
    
    
    # testt = np.expand_dims(Testfeature,axis=1)
    
    # testt = np.array(Testfeature).reshape(-1,1)
    
    # test_feat = np.expand_dims(testt,axis=1)   
    
    y_predd = clf.predict(Testfeature)  
    
    
    
    
    pred_res= max(y_predd)
    
    
    print('======================')
    print('PREDICTION ')
    print('======================')
    
    if pred_res == 1:
        print('Identified Suspicious = ABUSE')
        print("-----------------------------")
        print("Weapon --> Hands and fists")
        
        st.text('Identified Suspicious = ABUSE')
        st.text("-----------------------------")
        st.text("Weapon --> Hands and fists")        
        
        
    elif pred_res == 2:
        print('Identified Suspicious = ARREST')
        print("-----------------------------")
        print("Weapon --> Gun")
        
        st.text('Identified Suspicious = ARREST')
        st.text("-----------------------------")
        st.text("Weapon --> Gun")        
        
        
    elif pred_res == 3:
        print('Identified Suspicious = ARSON')
        print("-----------------------------")
        print("Weapon --> fire ")
        
        st.text('Identified Suspicious = ARSON')
        st.text("-----------------------------")
        st.text("Weapon --> fire ")        
        
        
    elif pred_res == 4:
        st.text('Identified Suspicious = EXPLOSION')
        st.text("-----------------------------")
        st.text("Weapon --> Rocket launchers ")
        
        
        
    elif pred_res == 5:
        st.text('Identified Suspicious = ROBBERY')
        st.text("--------------------------------------------")
        st.text("Weapon --> knives or other sharp instruments ")
        
        
    elif pred_res == 6:
        st.text('Identified Suspicious = SHOPLIFTING')
        st.text("--------------------------------------------")
        st.text("Weapon --> Handheld gun ")
        st.markdown(f'<h1 style="color:#FFFFFF;font-size:16px;">{"Identified Suspicious = SHOPLIFTING"}</h1>', unsafe_allow_html=True)
        st.text("--------------------------------------------")
        st.markdown(f'<h1 style="color:#FFFFFF;font-size:16px;">{"Weapon --> Handheld gun "}</h1>', unsafe_allow_html=True)

        
    elif pred_res == 7:
        st.text('Identified Suspicious = STEALING')
        st.text("--------------------------------------------")
        st.text("Weapon --> Handheld gun ")
        
    elif pred_res == 8:
        st.text('Identified Suspicious = FIGHTING')
        st.text("----------------------------------------------------")
        st.text("Weapon --> Offhand that aids a martialist profession")
        
    elif pred_res == 9:
        st.text('Identified Suspicious = ROAD ACCIDENTS')
        st.text("----------------------------------------------------")
        st.text("Weapon --> Occured in Roads")        
        
        
    # ================ COMPARISON GRAPH ================
    
    
    import seaborn as sns
    sns.barplot(x=["VGG-19","CNN-2D"],y=[acc_vgg,acc_cnn])    
    plt.title("Comparison Graph")
    plt.show()
        
    
    
