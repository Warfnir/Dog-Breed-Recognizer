from contextlib import redirect_stdout

import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# Resize all images from given path and save in other place.
# for i, image in enumerate(glob.glob('..\\Dog Breed Recognizer\\Images\\n02106662-German_shepherd\\*.jpg')):
#     im = Image.open(image)
#     print(im)
#     im = im.resize((150, 150))
#     print(im)
#     im.save(f'..\\Dog Breed Recognizer\\German_shepherds\\German_shepherd{i}.jpg')

# Creating model
model = Sequential()  # Model type
model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(120, 120, 3)))  # Convolutional layer with MaxPooling, looks for 32 filters with dim 3x3
model.add(MaxPooling2D(pool_size=(2,
                                  2)))  # Takes the max value from matrix 2x2 applied to the image.
# It's moved by 1px right till reaches max width and then
# 1px down till reaches max hight.

# Second layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# third layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# fourth layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# last convolutional layer
model.add(Conv2D(200, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattens the output -> from 2D or 3D data input makes 1D array.
model.add(Flatten())

# basic layer -> regular densely-connected NN
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))  # Helps preventing overfitting.

# last layer -> output layer
model.add(Dense(11, activation='softmax'))

# Here is defined the optimizer
# opt = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

BATCH_SIZE = 11  # Number of samples in mini batch, max is number of all samples.
STEPS_PER_EPOCH = 150  # Number of batches before an epoch is considered finished.
VALIDATION_STEPS = 50  # Number of batches but for validation.
EPOCHS = 100  # Number of epochs.

# Train data generator, transforms each image and prepares dataset
train_data_generator = ImageDataGenerator(rescale=1.0 / 255.0, zoom_range=0.2, horizontal_flip=True)
train_generator = train_data_generator.flow_from_directory('..\\DogBreedsDataset\\Train',
                                                           batch_size=BATCH_SIZE,
                                                           class_mode='categorical', target_size=(120, 120))
# Test data generator. Consist of images that weren't used in training.
# Doesn't transform them except for normalizing values.
test_data_generator = ImageDataGenerator(rescale=1.0 / 255.0)
validation_generator = test_data_generator.flow_from_directory('..\\DogBreedsDataset\\Test',
                                                               batch_size=BATCH_SIZE,
                                                               class_mode='categorical', target_size=(120, 120))

# Begin to train NN
# model.fit_generator(train_generator, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
#                     validation_data=validation_generator, validation_steps=VALIDATION_STEPS)

history = model.fit_generator(train_generator, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                              validation_data=validation_generator, validation_steps=VALIDATION_STEPS)

name = "FINALL"
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(name + "_ACC")

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(name + "_LOSS")
# model.summary()

with open(name + '_SUMMARY.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

model.save("breeds.h5")
# plot_model(model, to_file='model.png')
