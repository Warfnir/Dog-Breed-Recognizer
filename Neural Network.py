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
                 input_shape=(150, 150, 3)))  # Convolutional layer with MaxPooling, looks for 32 filters with dim 3x3
model.add(MaxPooling2D(pool_size=(2,
                                  2)))  # Takes the max value from matrix 2x2 applied to the image.
# It's moved by 1px right till reaches max width and then
# 1px down till reaches max hight.

# Second layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattens the output -> from 2D or 3D data input makes 1D array.
model.add(Flatten())

# Fourth layer -> regular densely-connected NN
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # Helps preventing overfitting.

# Fifth layer -> output layer
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

BATCH_SIZE = 8

# Train data generator, transforms each image and prepares dataset
train_data_generator = ImageDataGenerator(rescale=1.0 / 255.0, zoom_range=0.2, horizontal_flip=True)
train_generator = train_data_generator.flow_from_directory('..\\Dog Breed Recognizer\\data\\train',
                                                           batch_size=BATCH_SIZE,
                                                           class_mode='binary', target_size=(150, 150))
# Test data generator. Consist of images that weren't used in training.
# Doesn't transform them except for normalizing values.
test_data_generator = ImageDataGenerator(rescale=1.0 / 255.0)
validation_generator = test_data_generator.flow_from_directory('..\\Dog Breed Recognizer\\data\\validation',
                                                               batch_size=BATCH_SIZE,
                                                               class_mode='binary', target_size=(150, 150))

# Begin to train NN
model.fit_generator(train_generator, steps_per_epoch=2000 // BATCH_SIZE, epochs=10,
                    validation_data=validation_generator, validation_steps=800 // BATCH_SIZE)
