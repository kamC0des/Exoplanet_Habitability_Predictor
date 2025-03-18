import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def pre_model():
    # Define the API URL (CSV format for easier parsing)
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,hostname,pl_rade,pl_masse,pl_orbper,pl_eqt,st_spectype,st_teff,disc_pubdate,pl_orbsmax+from+ps&format=csv"

    #Fetching data
    '''response = requests.get(url)
    if response.status_code == 200:
        print("this worked")
        #Writing in binary "wb" to avoid any possible encoding issue or newline discrepancies.
        with open("exoplanets.csv", 'wb') as file:
            file.write(response.content)'''

    #Load the written file to pandas
    df = pd.read_csv(url)

    # df["disc_pubdate"] = pd.to_datetime(df["disc_pubdate"])
    # Drop duplicate exoplanet entries based on 'pl_name', keeping the most recent one
    df = df.sort_values(by="disc_pubdate", ascending=False).drop_duplicates(subset="pl_name", keep="first")

    #print(df.head())


    # Drop rows with missing values
    df = df.dropna()
    # print(df)

    #Establishing a habitable label for each planet based on equilibrium temperature, orbital max distance and stellar temperature

    df["habitable"] = (
        (
            (df["pl_eqt"].between(250, 350)) &
            #(df["pl_rade"].between(0.5, 2.0)) &
            (df["pl_orbsmax"].between(0.5, 1.5)) &
            (df["st_teff"] >= 5000)
        )  # Hotter stars with standard habitable orbits
        |
        (
            (df["pl_eqt"].between(250, 350)) &  # Include equilibrium temperature for cooler stars
            #(df["pl_rade"].between(0.5, 2.0)) &  # Ensure it's an Earth-sized planet
            (df["pl_orbsmax"].between(0.05, 0.3)) &
            (df["st_teff"] < 4000)
        )  # Cooler stars with closer planets
    ).astype(int)

    '''df_new = df[df["habitable"] == 1]
    print(df_new)'''
    return df

def split_train_and_test(df):
    train_data , test_data = train_test_split(df, test_size = 0.2, random_state = 42)
    #three features and one label for each exoplanet entry
    train_data = train_data[["pl_name", "pl_eqt", "st_teff", "pl_orbsmax", "habitable"]]
    test_data = test_data[["pl_name", "pl_eqt", "st_teff", "pl_orbsmax", "habitable"]]
    return train_data, test_data

def normalize(df):

    pl_eqt_max = np.array(df["pl_eqt"]).max()
    pl_eqt_min = np.array(df["pl_eqt"]).min()

    st_teff_max = np.array(df["st_teff"]).max()
    st_teff_min = np.array(df["st_teff"]).min()

    upper_limit = df["pl_orbsmax"].quantile(0.95)
    df["pl_orbsmax_clipped"] = df["pl_orbsmax"].clip(upper=upper_limit)


    df["pl_eqt"] = (np.array(df["pl_eqt"]) - pl_eqt_min)/(pl_eqt_max - pl_eqt_min)
    df["st_teff"] = (np.array(df["st_teff"]) - st_teff_min)/(st_teff_max - st_teff_min)
    #Used the MinMaxscaler() library because i clipped the 95% quantile of the orbsmax series to eliminate large outliers.
    #Could have used the MinMaxScaler() on the other features too but oh well
    scaler = MinMaxScaler()
    df["pl_orbsmax"] = scaler.fit_transform(df[["pl_orbsmax_clipped"]])

    return df

#<!-----------------------------Model Functions--------------------------------------------------------------------------------------------------------->

def relu(x):
    return np.maximum(0, x) # ReLU is a nn activation function that sets negative values to 0
def relu_derivative(x):
    return (x > 0).astype(float) # Gradient of ReLU (1 if x > 0, else 0). It is literally the derivative of ReLu
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # Sigmoid maps values to (0, 1) very important for our binary classification and will be used in the final output layer
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

#Network Structure
INPUT_SIZE = 3
HIDDEN_SIZE = 2
OUTPUT_SIZE = 1

W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.1 #Weight matrix for input -> hidden
b1 = np.zeros((1, HIDDEN_SIZE)) #Bias for the hidden layer. #Biases prevent neurons from being stuck as zero values/help shift activation values. Like weight, they are learned via backpropagation
W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.1
b2 = np.zeros((1, OUTPUT_SIZE)) #Bias for the output layer

def neural_network_training(iterations=1000, learning_rate=0.01):
    df = pre_model()
    df = normalize(df)
    train_data, test_data = split_train_and_test(df)
    X = np.array(train_data[["pl_eqt","st_teff","pl_orbsmax"]])
    Y = np.array(train_data["habitable"])
    #X has shape 140,3
    #Y has shape 140,
    '''print(X.shape)
    print(Y.shape)'''
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X)
        loss = compute_loss(Y, A2)
        backward_propagation(X, Y, Z1, A1, Z2, A2)

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    return test_data
def forward_propagation(X):
    global W1, b1, W2, b2
    # Compute activations
    Z1 = np.dot(X, W1) + b1 # produces a (140,2) matrix meaning all 140 training rows map to two hidden neurons
    A1 = relu(Z1) # Activation function at hidden layer
    Z2 = np.dot(A1, W2) + b2 #produces a (140,1) matrix meaning all 140 training rows map to one output neuron
    A2 = sigmoid(Z2) #Activation function at output layer to determine 0 or 1 for each entry which corresponds to habitable/not

    return Z1, A1, Z2, A2
def compute_loss(Y, A2):
    #We are using the Binary Cross-Entropy Loss function:
    # L = -1/N * Summation[ylog(y_hat) + (1-y)log(1-y_hat)]
    # where y = actual label matrix and y_hat = the nn output matrix
    # -1/N * Summation = mean of course
    loss = -np.mean(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return loss
def backward_propagation(X, Y, Z1, A1, Z2, A2, learning_rate=0.01):
    global W1, b1, W2, b2
    m = X.shape[0] #140 #number of exoplanets

    #compute gradients
    dZ2 = A2 - Y.reshape(m,1) # derivative of loss w.r.t Z2
    dW2 = np.dot(A1.T, dZ2) / m # Gradient of W2 (140,2) transpose = (2,140) * (140,1) == (2,1) = same shape as weight 2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m #Gradient of b2

    dZ1 = np.dot(dZ2, W2.T) * relu_derivative(Z1) # Backdrop through ReLU #dZ2 = (140, 1)  W2 = (1, 2) dot product == (140, 2)
    dW1 = np.dot(X.T, dZ1) / m #Gradient of W1 #X.T = (3,140) dZ1 = (140,2) dot product = (3,2)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m  #Gradient of b1

    #Update weights and biases using Gradient Descent
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

def neural_network_testing(test_data):
    X_test = np.array(test_data[["pl_eqt","st_teff","pl_orbsmax"]])
    Y_test = np.array(test_data["habitable"])

    Z1, A1, Z2, A2 = forward_propagation(X_test)
    hab_prediction = (A2 > 0.5).astype(int)
    test_data["habitable_prediction"] = hab_prediction
    print(test_data[["pl_name", "habitable", "habitable_prediction"]])
    loss = compute_loss(Y_test, A2)

    print(f"Test Loss: {loss:.4f}")
    print()
    print(f"The neural network has a test accuracy of {(1-loss)*100:.2f}%")




test_data = neural_network_training(iterations=1000, learning_rate=0.01)
neural_network_testing(test_data)

