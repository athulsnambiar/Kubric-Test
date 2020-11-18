import requests
import pandas
import scipy
import numpy as np
import numpy
import sys
import matplotlib.pyplot as plt 

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    csv_data = response.content.decode('utf-8').splitlines()
    # area = csv_data[0].split(',')
    
    area_model = np.array(list(map(float, csv_data[0].split(',')[1:])))

    price_model = np.array(list(map(float, csv_data[1].split(',')[1:])))
    # data_model = np.column_stack((area, price))
    ones_model = np.ones(area_model.shape)
    ar_sqrt_model = np.sqrt(area_model)
    ar_sqare_model = np.square(area_model)
    X_model = np.column_stack((area_model, ar_sqrt_model, ones_model))
    temp_model = np.linalg.pinv((np.matmul(X_model.T, X_model)))
    weights = np.matmul(np.matmul(temp_model, X_model.T), price_model)

    # print(weights)
    ar_sqrt = np.sqrt(area)
    ar_sqare = np.square(area)
    ones = np.ones(area.shape)
    X = np.column_stack((area, ar_sqrt, ones))
    # print(X.shape)
    # print(weights.shape)
    y = np.matmul(X, weights)
    # print(y.shape)
    

    

    # print(price)
    # print(data)
    # plt.scatter(area_model, price_model, c ="blue")
    # plt.scatter(area, y, c ="red")
    
  
    # plt.show() 
    return y
    


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
