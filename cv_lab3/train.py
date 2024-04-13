
 
from keras.layers import Input,Dense  
 
from keras.layers import Flatten,Lambda,Dropout  
 
from keras.models import Model  
 
import keras.backend as K  
 
from keras.models import load_model  
 
import numpy as np  
 
from PIL import Image  
 
import glob  
 
import matplotlib.pyplot as plt  
 
from PIL import Image  
 
import random  
 
from keras.optimizers import Adam,RMSprop  
 
import tensorflow as tf  
 
def create_base_network(input_shape):  
 
    image_input = Input(shape=input_shape)  
 
    x = Flatten()(image_input)  
 
    x = Dense(128, activation='relu')(x)  
 
    x = Dropout(0.1)(x)  
 
    x = Dense(128, activation='relu')(x)  
 
    x = Dropout(0.1)(x)  
 
    x = Dense(128, activation='relu')(x)  
 
    model = Model(image_input,x,name = 'base_network')  
 
    return model  
 
def contrastive_loss(y_true, y_pred):  
 
     margin = 1  
 
     sqaure_pred = K.square(y_pred)  
 
     margin_square = K.square(K.maximum(margin - y_pred, 0))  
 
     return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)  
 
def accuracy(y_true, y_pred): # Tensor上的操作  
 
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))  
 
def siamese(input_shape):  
 
    base_network = create_base_network(input_shape)  
 
    input_image_1 = Input(shape=input_shape)  
 
    input_image_2 = Input(shape=input_shape)  
 
  
 
    encoded_image_1 = base_network(input_image_1)  
 
    encoded_image_2 = base_network(input_image_2)  
 
  
 
    l2_distance_layer = Lambda(  
 
        lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True))  
 
        ,output_shape=lambda shapes:(shapes[0][0],1))  
 
    l2_distance = l2_distance_layer([encoded_image_1, encoded_image_2])  
 
      
 
    model = Model([input_image_1,input_image_2],l2_distance)  
 
      
 
    return model  
 
def process(i):  
 
    img = Image.open(i,"r")  
 
    img = img.convert("L")  
 
    img = img.resize((wid,hei))  
 
    img = np.array(img).reshape((wid,hei,1))/255  
 
    return img  
 
#model = load_model("testnumber.h5",custom_objects={'contrastive_loss':contrastive_loss,'accuracy':accuracy})  
 
wid=28  
 
hei=28  
 
model = siamese((wid,hei,1))  
 
imgset=[[],[],[],[],[],[],[],[],[],[]]  
 
for i in glob.glob(r"train_images\*.jpg"):  
 
    imgset[int(i[-5])].append(process(i))  
 
size = 60000  
 
  
 
r1set = []  
 
r2set = []  
 
flag = []  
 
for j in range(size):  
 
    if j%2==0:  
 
        index = random.randint(0,9)  
 
        r1 = imgset[index][random.randint(0,len(imgset[index])-1)]  
 
        r2 = imgset[index][random.randint(0,len(imgset[index])-1)]  
 
        r1set.append(r1)  
 
        r2set.append(r2)  
 
        flag.append(1.0)  
 
    else:  
 
        index1 = random.randint(0,9)  
 
        index2 = random.randint(0,9)  
 
        while index1==index2:  
 
            index1 = random.randint(0,9)  
 
            index2 = random.randint(0,9)  
 
        r1 = imgset[index1][random.randint(0,len(imgset[index1])-1)]  
 
        r2 = imgset[index2][random.randint(0,len(imgset[index2])-1)]  
 
        r1set.append(r1)  
 
        r2set.append(r2)  
 
        flag.append(0.0)  
 
r1set = np.array(r1set)  
 
r2set = np.array(r2set)  
 
flag = np.array(flag)  
 
model.compile(loss = contrastive_loss,  
 
            optimizer = RMSprop(),  
 
            metrics = [accuracy])  
 
history = model.fit([r1set,r2set],flag,batch_size=128,epochs=20,verbose=2)  
 
# 绘制训练 & 验证的损失值  
 
plt.figure()  
 
plt.subplot(2,2,1)  
 
plt.plot(history.history['accuracy'])  
 
plt.title('Model accuracy')  
 
plt.ylabel('Accuracy')  
 
plt.xlabel('Epoch')  
 
plt.legend(['Train'], loc='upper left')  
 
plt.subplot(2,2,2)  
 
plt.plot(history.history['loss'])  
 
plt.title('Model loss')  
 
plt.ylabel('Loss')  
 
plt.xlabel('Epoch')  
 
plt.legend(['Train'], loc='upper left')  
 
plt.show()  
 
model.save("testnumber.h5")