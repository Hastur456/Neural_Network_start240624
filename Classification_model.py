from keras.api.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.api.models import Sequential


def get_classification_model(im_size):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(im_size, im_size, 1)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation="relu"),
        Dense(33, activation="softmax")
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
