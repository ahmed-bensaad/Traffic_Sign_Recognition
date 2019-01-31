import numpy as np 
from sklearn.base import BaseEstimator
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense,Dropout, Flatten,BatchNormalization
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import functools
import tensorflow as tf
import cv2 
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, Callback, LearningRateScheduler


class Classifier(BaseEstimator):
    def __init__(self, n_epochs = 15, batch_size = 64, lr = 1e-3):
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.model = create_model()

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2)
        ntrain_samples = X_train.shape[0]
        nval_samples = X_val.shape[0]
        validation_steps = nval_samples / self.batch_size 
        steps_per_epoch = 5 * ntrain_samples / self.batch_size # *5 for random data augmentation
        validation_steps = nval_samples / self.batch_size

        callbacks_list = []
        callbacks_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=5, min_delta=0.001,
                                        cooldown=2, verbose=1))

        self.model.fit_generator(train_generator(X_train ,y_train,self.batch_size),
                                steps_per_epoch=steps_per_epoch, epochs = self.epochs, 
                                validation_data =val_generator(X_val ,y_val,self.batch_size),
                                 validation_steps = validation_steps,
                                  callbacks = callbacks_list )
        

    def predict(self, X):
        imgs = [cv2.imread(path) for path in X]
        imgs_resized = [cv2.resize(img, (128,128), interpolation=cv2.INTER_LINEAR) for img in imgs]
        return self.model.predict_classes(np.array(imgs_resized))

    def predict_proba(self, X):
        imgs = np.array([cv2.imread(path) for path in X])
        imgs_resized = [cv2.resize(img, (128,128), interpolation=cv2.INTER_LINEAR) for img in imgs]
        return self.model.predict(np.array(imgs_resized))

####################### Model ###########################

def create_model(input_size = (128,128,3), epochs = 100 , lr = 1e-3 ):
    model = Sequential()
    model.add(Conv2D(3, input_shape= input_size, kernel_size=1, padding="same", activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, kernel_size=5, padding="same", activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(80, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(100, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(150, kernel_size=3, padding="valid", activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0,5))
    
    model.add(Dense(43, activation='softmax'))
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)


    model.compile(loss='categorical_crossentropy',optimizer = sgd, metrics=['accuracy'])
    

    
    return model


####################### Generators ###########################

def train_generator(anns_x, anns_y, batch_size = 64, target_size = (128,128), classes = 43):
    n_samples = anns_x.shape[0]
    while True:
        original_batch_size = batch_size // 8
        for i in range(0, n_samples, original_batch_size):
            imgs = [cv2.imread(file) for file in anns_x.iloc[i:i+(original_batch_size)]]
            imgs_resized = [cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR) for img in imgs]
            X_batch = [img for img in imgs_resized]
            Y_batch = list(anns_y.iloc[i:i+original_batch_size])

            #Data Augmentation : we will generate 8 images for each existing image
            for j in range(len(X_batch)):
                for _ in range(8):
                    X_batch.append(transform_image(X_batch[j]))
                    Y_batch.append(Y_batch[j])
            yield np.array(X_batch), to_categorical(np.array(Y_batch),num_classes=43)


def val_generator(anns_x, anns_y, batch_size = 100, target_size = (128,128), classes = 43):
  n_samples = anns_x.shape[0]
  while True:

      for i in range(0, n_samples, batch_size):
          imgs = [cv2.imread(file) for file in anns_x.iloc[i:i+batch_size]]
          imgs_resized = [cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR) for img in imgs]
          X_batch = [img for img in imgs_resized]
          Y_batch = anns_y.iloc[i:i+batch_size]
          yield np.array(X_batch), to_categorical(np.array(Y_batch),num_classes=43)


########################## Data Augmentation ##################################
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range = 35,
                     shear_range=10,trans_range = 10):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    
    # Brightness 
    

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    img = augment_brightness_camera_images(img)
    
    return img

