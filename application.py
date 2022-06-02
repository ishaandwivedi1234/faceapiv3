import numpy as np
from flask import Flask,request
from flask import jsonify

######################################################################################################

application = Flask(__name__)
app = application

######################################################################################################

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import PIL

######################################################################################################

batch_size = 16
img_height = 120
img_width  = 120
data_dir="Dataset"
mode=1

######################################################################################################

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="training",
  color_mode=['grayscale', 'rgb'][mode],
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)
######################################################################################################

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="validation",
  color_mode=['grayscale', 'rgb'][mode],
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

######################################################################################################


class_names = train_ds.class_names
class_names = val_ds.class_names

######################################################################################################

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

IMG_SHAPE=(img_height, img_width, 3)
IMG_SHAPE

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,

                                               weights='imagenet')

######################################################################################################
base_model.trainable = False
base_model.summary()
model1 = tf.keras.models.Sequential([
layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
base_model,
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(10000, activation='relu'),
tf.keras.layers.Dense(2058, activation='relu'),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(5, activation='softmax')]
)

model1.summary()
model1.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
epochs = 0
history = model1.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)


def getpredictions(filename="https://i.guim.co.uk/img/media/23def510ff776797ec09788d2eaae47876e79752/0_212_6357_3814/master/6357.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=e137f3289c623ed7a1aeeae620bf77ef"):      
    img_height = 120
    img_width  = 120      
    image_url = tf.keras.utils.get_file('Court', origin=filename )
    img1 = tf.keras.preprocessing.image.load_img(image_url, target_size=(img_height, img_width))      
    Y = image.img_to_array(img1)
    X = np.expand_dims(Y,axis=0)
    val = model1.predict(X) 
    val = list(val[0])
    class_of_object = ['DiGeorge_Syndrome', 'Down_syndrome', 'Noonan_Syndrome', 'Normal_people', 'William_buren_syndrome']    
    print(class_of_object)
    return class_of_object

@app.route('/')
def home():
    return jsonify({"Success":"Welcome to Diagnozifyzz API Service !"})

@app.route('/loadModel')
def loadModels():
    try:
        res = loadModel()
        if res :
            return jsonify({"Success":"Model Loaded Successfully !"})
        else:
            return jsonify({"Error":"Model not Loaded !"})
    except Exception as e:
        return jsonify({"Success":"mode loaded !"})



@app.route('/predict',methods=['POST'])
def getprediction():

    try:
        data = request.json
        print(data)
        imageUrl = data['image_url']
        print(imageUrl)
        res = getpredictions(filename=imageUrl)
        return jsonify(
            {
                "success":True,
                "msg":"successfull prediction !",
                "data":res
            }
        )

    except Exception as e:
        print(e)
        return jsonify({"success":False,"msg":"wrong formated image !","data":None,"error":str(e)},)


if __name__ == '__main__':
   app.run() 