import tensorflow as tf
import joblib, requests
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime


load_model = tf.keras.models.load_model



yf.pdr_override()

############################################
end = datetime.now()
start = datetime(end.year - 11, end.month, end.day)

#############################################################################



def selection(select):
    if select == "apple":
        model = load_model(r".\pages\models\apple_model.h5")
        scaler = joblib.load(r".\pages\models\apple_scaler.h5")
        df = yf.download("AAPL", start, end)

    elif select == "google":
        model = load_model(r".\pages\models\google_model.h5")
        scaler = joblib.load(r".\pages\models\google_scaler.h5")
        df = yf.download("GOOGL", start, end)

    elif select == "amazon":
        model = load_model(r".\pages\models\amazon_model.h5")
        scaler = joblib.load(r".\pages\models\amazon_scaler.h5")
        df = yf.download("AMZN", start, end)

    
    elif select == "microsoft":
        model = load_model(r".\pages\models\microsoft_model.h5")
        scaler = joblib.load(r".\pages\models\microsoft_scaler.h5")
        df = yf.download("MSFT", start, end)

    
    
    elif select == "meta":
        model = load_model(r".\pages\models\meta_model.h5")
        scaler = joblib.load(r".\pages\models\meta_scaler.h5")
        df = yf.download("meta", start, end)

    
    
    elif select == "netflix":
        model = load_model(r".\pages\models\netflix_model.h5")
        scaler = joblib.load(r".\pages\models\netflix_scaler.h5")
        df = yf.download("NFLX", start, end)



    elif select == "tesla":
        model = load_model(r".\pages\models\tesla_model.h5")
        scaler = joblib.load(r".\pages\models\tesla_scaler.h5")
        df = yf.download('TSLA', start, end)



    elif select == "intel":
        model = load_model(r".\pages\models\intel_model.h5")
        scaler = joblib.load(r".\pages\models\intel_scaler.h5")
        df = yf.download('INTC', start, end)



    elif select == "nvidia":
        model = load_model(r".\pages\models\nvidia_model.h5")
        scaler = joblib.load(r".\pages\models\nvidia_scaler.h5")
        df = yf.download("NVDA", start, end)


    return model, scaler , df


def predict(n, select):
    model, scaler, df = selection(select)
    if (n <= 7) & (n >= 1):
    
        num = 1
        last = []

        last_60_value = df["Close"].iloc[-60:]

        for i in last_60_value:
            last.append(i)


        for i in range(n):
            np_list = np.reshape(last[-60:], (-1, 1))
            last_csl = scaler.transform(np_list)
            last_csl = np.reshape(last_csl, (1, 60, 1))

            pred = model.predict(last_csl)
            prediction = float(scaler.inverse_transform(pred)[0][0])

            last.append(prediction)
            print(f"The prediction number {num} is : {prediction}")
            num = num + 1

        return last[60:]
    
    return "Sorry you must inter mostly seven days or 'week' or less"



