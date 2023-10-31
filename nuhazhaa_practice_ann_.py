#Teori AI
#Tugas 5 - Nerual Network Multiperceptron
#Siaga Whiky Setia - 1123800002 - S2 Teknik Elektro PENS

#import Libraries
import numpy as np
import matplotlib.pyplot as plt

#1. Define Function Aktivasi-nya (Sigmoid)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#2. Menentukan Parameter Awal
def parameter_awal(nilai_input, hidden_unit, nilai_output):
    W1 = np.random.randn(hidden_unit, nilai_input)
    W2 = np.random.randn(nilai_output, hidden_unit)
    b1 = np.ones((hidden_unit, 1))
    b2 = np.ones((nilai_output, 1))

    parameters = {"W1" : W1, "b1" : b1,
                  "W2" : W2, "b2" : b2}
    return parameters

#3. Menentukan Persamaan Feed_Forward
def forward(X, Y, parameters):
    x_in = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    #Tambahkan bias
    Z1 = np.dot(W1, X) + b1         

    #Fungsi Aktivasi
    A1 = sigmoid(Z1)

    #Tambahkan bias           
    Z2 = np.dot(W2, A1) + b2 

    #Fungsi Aktivasi       
    A2 = sigmoid(Z2)                

    stor = (Z1, A1, W1, b1, Z2, A2, W2, b2)
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    total = -np.sum(logprobs) / x_in
    return total, stor, A2

#4. Menentukan Persamaan Backward Propagation
def backward_propagation(X, Y, stor):
    x_in = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2) = stor

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / x_in
    db2 = np.sum(dZ2, axis = 1, keepdims = True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, A1 * (1 - A1))
    dW1 = np.dot(dZ1, X.T) / x_in
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / x_in

    gradients = {"dZ2" : dZ2, "dW2" : dW2, "db2" : db2,
                 "dZ1" : dZ1, "dW1" : dW1, "db1" : db1}
    return gradients

#5. Memperbarui nilai dari bobot dan bias
def pembaruanParameters(parameters, gradients, miu):
    parameters["W1"] = parameters["W1"] - miu * gradients["dW1"]
    parameters["W2"] = parameters["W2"] - miu * gradients["dW2"]
    parameters["b1"] = parameters["b1"] - miu * gradients["db1"]
    parameters["b2"] = parameters["b2"] - miu * gradients["db2"]
    return parameters

#6. Masukkan data training, dimana disini menggunakan input XOR
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) #XOR Input
Y = np.array([[0, 1, 1, 0]])               #XOR Output

#7. Menentukan model dari Parameters
hidden_unit = 2
nilai_input = X.shape[0]
nilai_output = Y.shape[0]
parameters = parameter_awal(nilai_input, hidden_unit, nilai_output)
epoch = 100000
miu = 0.01
losses = np.zeros((epoch, 1))
#mse_losses = np.zeros((epoch, 1))
#sse_losses = np.zeros((epoch, 1))

#8. Start Training
mse_values = []
sse_values = []

for i in range(epoch):
    losses[i, 0], stor, A2 = forward(X, Y, parameters)
    gradients = backward_propagation(X, Y, stor)
    parameters = pembaruanParameters(parameters, gradients, miu)
    
    # Menghitung MSE dan SSE di setiap iterasi
    mse = np.mean(np.square(Y - A2))
    sse = np.sum(np.square(Y - A2))

    mse_values.append(mse)
    sse_values.append(sse)
    
    # Mencetak MSE dan SSE di setiap iterasi
    print("Epoch:", i, " - Mean Squared Error (MSE):", mse, " - Sum of Squared Errors (SSE):", sse)

#9. Plot grafik performa, MSE, dan SSE
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(mse_values, color='blue', label='MSE')
plt.title("Graph Plot of MSE")
plt.xlabel("Epochs")
plt.ylabel("MSE Values")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sse_values, color='green', label='SSE')
plt.title("Graph Plot of SSE")
plt.xlabel("Epochs")
plt.ylabel("SSE Values")
plt.legend()

plt.show()

print("\nFinal Mean Squared Error (MSE):", mse)
print("Final Sum of Squared Errors (SSE):", sse)

#10. Testing Data menggunakan Data Training
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) #XOR Input
total, _, A2 = forward(X, Y, parameters)

prediction = (A2 > 0.5) * 1.0
# print(A2)
np.set_printoptions(precision=4)
print(f"\nPrediksi Nilai Output dengan Data Training :", prediction)