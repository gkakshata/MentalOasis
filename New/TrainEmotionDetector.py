# import required packages
import cv2
from keras.models import Sequential #user-friendly interface to build, train, and deploy deep learning models.
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling to use for preprocessing
train_data_gen = ImageDataGenerator(rescale=1./255)#for training
validation_data_gen = ImageDataGenerator(rescale=1./255)#for testing/validation

# Preprocess all test images using flow_from_directory tool of keras
train_generator = train_data_gen.flow_from_directory(
        'data/train',#target folder
        target_size=(48, 48),#target size
        batch_size=64,#number of samples processed before the model is updated
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# create model structure
emotion_model = Sequential()#model in keras

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
# Add a 2D convolutional layer with 32 filters, a kernel size of (3, 3), ReLU activation, and an input shape of (48, 48, 1)
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# Add another 2D convolutional layer with 64 filters and a kernel size of (3, 3), using ReLU activation
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# Add a max pooling layer with a pool size of (2, 2)
emotion_model.add(Dropout(0.25))
# Add a dropout layer with a rate of 0.25 to prevent overfitting
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# Add another 2D convolutional layer with 128 filters and a kernel size of (3, 3), using ReLU activation
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# Add another max pooling layer with a pool size of (2, 2)
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# Add another 2D convolutional layer with 128 filters and a kernel size of (3, 3), using ReLU activation
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# Add another max pooling layer with a pool size of (2, 2)
emotion_model.add(Dropout(0.25))
# Add another dropout layer with a rate of 0.25
emotion_model.add(Flatten())
# Flatten the output of the previous layer to a 1D vector
emotion_model.add(Dense(1024, activation='relu'))
# Add a fully connected layer with 1024 units and ReLU activation
emotion_model.add(Dropout(0.5))
# Add another dropout layer with a rate of 0.5
emotion_model.add(Dense(7, activation='softmax'))
# Add a final fully connected layer with 7 units (corresponding to the 7 emotions) and softmax activation

cv2.ocl.setUseOpenCL(False)
# Disable OpenCL usage in OpenCV to avoid compatibility issues


emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
# Compile the model with categorical cross-entropy as the loss function,Adam optimizer with a 
# learning rate of 0.0001 and decay rate of 1e-6,and accuracy as the evaluation metric.

# Train the neural network/model
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,#total no of images divided by 64
        epochs=50,#the training of the neural network with all the training data for one cycle.
        validation_data=validation_generator,
        validation_steps=7178 // 64)

# save model structure in jason file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.h5')

