import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import Flatten, Dense 
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam,SGD 
from matplotlib.patches import Rectangle
from imageio import imread 
ch = imread('sho.png')

print(type(ch),ch.shape)

outDim = 500
dog = np.array(ch)
h,w,_ = dog.shape
 
def imageGenerator(batch_size=5):
    #generate batches of images and targets
    while True:
        #each epoch has no_of_batches 
        for _ in range(50):
            X = np.zeros((batch_size,outDim,outDim,3))
            Y = np.zeros((batch_size,4))

            for i in range(batch_size):
                # make the boxes and store their location in target 
                row0 = np.random.randint(outDim-h)
                col0 = np.random.randint(outDim-w)
             #atleast 10 size ke box ka scope dene ke liye humne yaha 90 liya h 
                row1 = row0+h
                col1 = col0+w
                X[i,row0:row1, col0:col1,:] = dog[:,:,:3]
                Y[i,0] = row0/outDim 
                Y[i,1] = col0/outDim 
                Y[i,2] = (row1-row0)/outDim
                Y[i,3] = (col1-col0)/outDim

            yield X/255,Y

def makeModel():
    vgg = tf.keras.applications.VGG16(
        input_shape=[outDim,outDim,3],include_top=False,weights='imagenet')

    x = Flatten()(vgg.output)
    x = Dense(4,activation='sigmoid')(x)
    model = Model(vgg.input,x)

    model.compile(loss='binary_crossentropy',optimizer = 'adam')
    return model 
model = makeModel()
model.fit(imageGenerator(),steps_per_epoch=50,epochs = 5)

def plot_preds():
    x = np.zeros((outDim,outDim,3))
    row0 = np.random.randint(outDim-h)
    col0 = np.random.randint(outDim-w)
    row1 = row0+h
    col1 = col0+w
    x[row0:row1,col0:col1,:]= dog[:,:,:3]
    X = np.expand_dims(x,0)/255

    y = np.zeros(4)
    y[0] = row0/outDim
    y[1] = col0/outDim
    y[2] = (row1-row0)/outDim
    y[3] = (col1-col0)/outDim
    
    p = model.predict(X)[0]
    fig,ax= plt.subplots(1)
    ax.imshow(x)
    rect = Rectangle(
        (p[1]*outDim,p[0]*outDim ),
        p[3]*outDim,p[2]*outDim,linewidth =1,edgecolor='r'    )
    ax.add_patch(rect)
    plt.show()

    

plot_preds()