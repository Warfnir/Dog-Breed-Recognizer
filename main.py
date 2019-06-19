import numpy as np
from keras.models import load_model
from keras.preprocessing import image

try:
    # load breeds
    file = open('breeds.txt', 'r')
    names = file.read()
    decoder = names.splitlines()
    print(decoder)

    # load image as input
    img = image.load_img('image.jpg', target_size=(120, 120))
    x = image.img_to_array(img)
    x = np.array([x, ])

    # load network model and compile
    model = load_model('FINAL.h5')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # predict result
    result = model.predict_classes(x)
    print("\n\nYour result is...:\n" + decoder[result[0]])
except Exception as e:
    print(e)
finally:
    input("Press enter to exit...")
