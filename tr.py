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
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
# filename = askopenfilename()
Frames = [] 
Frames_all = []
for fitr in range(0,10):

    path = 'Dataset/V('
    cap = cv2.VideoCapture(path+str(fitr)+').mp4')
     
     
    # Check if camera opened successfully
    if (cap.isOpened()== False):
      print("Error opening video stream or file")
    else:
          
        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
         
                # Display the resulting frame
                cv2.imshow('Frame', frame)
                Frames.append(frame)
                Frames_all.append(frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
         
            # Break the loop
            else:
                break
         
        # When everything done, release the video capture object
        cap.release()
         
        # Closes all the frames
        cv2.destroyAllWindows()
        
    from PIL import Image 
    import PIL 
    # for iii in range(0,len(Frames)):

        # mpimg.imsave(str(fitr+1)+'/'+str(iii+1)+'.jpg',Frames[iii]) 
# === GETTING INPUT


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
Train_Features_7 = []
for iiij in range(0,len(Frames_all)):
    
    img1 = Frames_all[iiij]
    
    # plt.imshow(img1)
    # plt.title('ORIGINAL FACE IMAGE')
    # plt.show()
    
    
    # PRE-PROCESSING
    
    h1=512
    w1=512
    
    dimension = (w1, h1) 
    resized_image1 = cv2.resize(img1,(h1,w1))
    
    # fig = plt.figure()
    # plt.title('RESIZED FACE IMAGE')
    # plt.imshow(resized_image1)
    # plt.show()
    
    
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    
    # plt.imsave('RoadAcc/'+str(iiij)+'.jpg',resized_image1 )
    
    # ==========================================================================
    
    # FACE Detection
    
    # face_cascade = cv2.CascadeClassifier('opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
    # #eye_cascade = cv2.CascadeClassifier('opencv-master\data\haarcascades\haarcascade_eye.xml')
    
    # #face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    # #eye_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')
    
    # #face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.4.0_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    # #eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.4.0_1/share/OpenCV/haarcascades/haarcascade_eye.xml')
    
    # #imgg = cv2.imread('Dataset\1.jpg')
    
    # img_face = resized_image1
    # grayscale = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
    
    # faces = face_cascade.detectMultiScale(grayscale, 1.3, 2)
    
    # for (x,y,w,h) in faces:
        
    #     img_face = cv2.rectangle(img_face,(x,y),(x+w,y+h),(255,0,0),2)
        
    #     roi_gray = grayscale[y:y+h, x:x+w]
    #     roi_color = img_face[y:y+h, x:x+w]
        
    
    # # fig = plt.figure()
    # # plt.imshow(img_face)
    # # plt.show()
    
    from skimage.feature import greycomatrix, greycoprops
    
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
        glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,symmetric=True)
        xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(greycoprops(glcm, 'correlation')[0, 0])
    
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
    
    Train_Features_7.append(Glcm_fea2)
import pickle
with open('Train_Features_7.pickle', 'wb') as f:
    pickle.dump(Train_Features_7, f) 


    # Importing Image module from PIL package 
   

    # Frames[iii].save(str(iii+1)+'.jpg')
        