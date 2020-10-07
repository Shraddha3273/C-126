import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml as fol
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from PIL import Image
import PIL.ImageOps

# fetch dataset from openml
X,y = fol('mnist_784', version=1, return_X_y=True)
X_train, X_test, y_train, y_test = tts(X,y, random_state=33, train_size=7500, test_size=2500)
X_train_scaled = X_train/255
X_test_scaled = X_test/255

modelData = lr(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

# function to predict the data
def get_prediction(image) : 
    impil = Image.open(image)
    imbw = impil.convert('L')
    imbw_resize = imbw.resize((28,28), Image.ANTIALIAS)
    px_filter = 20
    min_px = np.percentile(imbw_resize, px_filter)
    imbw_inverted = np.clip(imbw_resize-min_px, 0,255)
    max_px = np.max(imbw_resize)
    imbw_inverted = np.asarray(imbw_inverted)/max_px
    test_sample = np.array(imbw_inverted).reshape(1,784)
    test_predict = modelData.predict(test_sample)
    return test_predict[0]