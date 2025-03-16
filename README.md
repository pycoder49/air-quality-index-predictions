# air-quality-index-predictions
Implementing a neural network allows customizability for the number of hidden layers and neurons in each layer.
Implemented Adam optimizer using both momentum and velocities for weights and biases.

Used OpenWeather Air-Pollution API to get data from Jan 1st, 2015 to Jan 1st, 2025.

Trained the model over the data.

### Logistics:
1. From OpenWeather API:
   - Air quality index ranges from 1-5
   - This project follows this encoding:
     - 0 (good):     1-2
     - 1 (moderate): 3-4
     - 2 (bad):      5+
   - This is further encoded using one hot encoding
     - [1, 0, 0] = 0
     - [0, 1, 0] = 1
     - [0, 0, 1] = 2
2. Model accuracy post-training: 92.03%
   - Layout used: Input, HL1, HL2, HL3, Output
   - Neuron count: 8, 20, 15, 10, 3
3. Loss over time

![image](https://github.com/user-attachments/assets/799f2025-eda4-4aa6-bf33-1ce81dffeae7)

![loss](https://github.com/user-attachments/assets/9c26ceb7-e4da-4c79-8663-2f4f42c81670)
