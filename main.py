import cv2 as cv
import keras
import numpy as np
import matplotlib.pyplot as plt


def image_convert(img):
    img = cv.imread(img, 0)
    img = cv.cvtColor(img, cv.COLOR_BAYER_RG2GRAY)
    return img


def find_conturs(img):
    img = image_convert(img)
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    t, thresh_img = cv.threshold(blurred, 215, 255, cv.THRESH_BINARY)
    thresh_img = 255 - thresh_img
    conts, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return conts


def get_letters_images(img):
    conturs = find_conturs(img)
    img = image_convert(img)

    letter_images = []
    for idx, contur in enumerate(conturs):
        x, y, w, h = cv.boundingRect(contur)
        letter_image = img[y:y + h, x:x + w]
        letter_image = cv.cvtColor(letter_image, cv.COLOR_BAYER_RG2GRAY)
        letter_images.append((x, w, letter_image))

    letter_images.sort(key=lambda n: n[0])
    return letter_images


def show_symbols(img):
    fig, ax = plt.subplots(10, 3, figsize=(5, 10))
    fig.subplots_adjust()
    for axi, image in zip(ax.flat, get_letters_images(img)):
        axi.imshow(image[2], cmap='gray')
    fig.show()
    fig.waitforbuttonpress()


def predict_classification_model(model: any, letter_image, classes: list, im_size=64):
    image_ = cv.copyMakeBorder(letter_image, 10, 10, 10, 10,
                               borderType=cv.BORDER_CONSTANT, value=[255])
    image_ = cv.resize(image_, (im_size, im_size), interpolation=cv.INTER_AREA)
    image_ = cv.cvtColor(image_, cv.COLOR_BAYER_GR2GRAY)
    image_ = keras.utils.img_to_array(image_)
    image_ = np.expand_dims(image_, axis=0)

    predict_ = model.predict(image_)
    result = np.argmax(predict_, axis=1)
    return classes[int(dataset_classes[result[0]])]


def image_to_string(img, model: any, classes: list):
    letters = get_letters_images(img)

    string_out = ""
    for i in range(len(letters)):
        dn = letters[i + 1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        string_out += predict_classification_model(model, letters[i][2], classes)
        if dn > letters[i][1]/3:
            string_out += " "
    return string_out


model_ = keras.models.load_model("Models\\Model3.keras")

image_hello = "Test_Images/Hello.png"
image_pain = "Test_Images/Pain2.png"

alphabet = ["а","б","в","г","д","е","ё","ж","з","и","й","к","л","м","н","о",
            "п","р","с","т","у","ф","х","ц","ч","ш","щ","ъ","ы","ь","э","ю","я"]
dataset_classes = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20',
                   '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '4',
                   '5', '6', '7', '8', '9']

print(image_to_string(image_hello, model_, alphabet))
print(image_to_string(image_pain, model_, alphabet))
