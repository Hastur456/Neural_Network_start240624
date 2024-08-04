import keras.api.utils
from keras.api.preprocessing import image_dataset_from_directory
import Classification_model


data_directory = "Dataset\\Train"
im_size = 64

train_data = image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="training",
    image_size=(im_size, im_size),
    seed=123,
    color_mode="grayscale")

validation_data = image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="validation",
    image_size=(im_size, im_size),
    seed=123,
    color_mode="grayscale")

print(train_data.class_names)

model = Classification_model.get_classification_model(im_size)

model.fit(x=train_data, epochs=10, validation_data=validation_data, validation_steps=20)
model.save("Model4.keras")


