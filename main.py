from datetime import datetime
from NeuralNetwork import NeuralNetwork
from dotenv import load_dotenv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch
import WeatherData

load_dotenv()

# constants
APPID = os.getenv("APPID")
START_DATE = int(datetime(2015, 1, 1).timestamp())  # Jan 1st, 2015
END_DATE = int(datetime(2025, 1, 1).timestamp())  # Jan 1st, 2025


def get_data() -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :return: datasets --> [x_train, y_train, x_test, y_test]
    """
    # getting data
    data = WeatherData.fetch_data(START_DATE, END_DATE, APPID)
    # obtaining datasets
    datasets = WeatherData.clean_data(data)
    # datasets = [x_train, y_train, x_test, y_test]
    return datasets


def encode(y: torch.Tensor) -> torch.Tensor:
    """
    Categories: 0 --> good (1-2)
                1 --> moderate (3-4)
                2 --> bad (5+)

    First, maps the y tensor values to respective categories
    Then, returns one hot encoded mapped tensor

    :param y: tensor with ranges 1-5 -- weather aqi value
    :return: one hot encoded tensor
    """
    y = y.squeeze(-1).long()

    mapping = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}

    y_mapped = torch.zeros_like(y)
    for i, value in enumerate(y):
        y_mapped[i] = mapping[value.item()]

    y_one_hot = F.one_hot(y_mapped, num_classes=3).float()
    """
    1,0,0 = 0
    0,1,0 = 1
    0,0,1 = 2
    
    example one-hot encoded tensor:
    tensor([[1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            ...,
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 0., 0.]])
    """

    return y_one_hot


def get_accuracy(predictions: torch.Tensor, target: torch.Tensor) -> float:
    correct_count = torch.all(predictions == target, dim=1).sum().item()
    return (correct_count / target.shape[0]) * 100


def plot_training_loss(losses):
    """
    Plots the training loss curve.

    :param losses: List of loss values recorded during training.
    """
    plt.plot(losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.legend()
    plt.show()


def main():
    """
    :return: None
    """

    """ getting data """
    datasets = get_data()

    x_train = datasets[0]
    y_train = datasets[1]
    x_test = datasets[2]
    y_test = datasets[3]

    # getting encoded y_train and y_test
    y_one_hot_train = encode(y_train)
    y_one_hot_test = encode(y_test)

    """ training loop """
    network = NeuralNetwork(20, 15, 10, input_size=8, output_size=3, learning_rate=0.1)
    # network stats:
    # Input layer:      8
    # Hidden layer 1:   15
    # Hidden layer 2:   10
    # Hidden layer 3:   5
    # Output layer:     3

    epochs = 1000
    losses = []
    for epoch in range(epochs):
        # forward pass
        predictions = network.forward(x_train)

        # computing loss
        loss = network.calculate_loss(predictions, y_one_hot_train)
        losses.append(loss.item())

        # backward pass
        network.backprop(x_train, y_one_hot_train)

        # updating weights and biases using adam optimizer
        network.update_parameters(adam_step=True)

        # printing loss for tracking -- every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch} loss: {loss}")
        elif epoch == epochs - 1:
            print(f"Epoch {epochs} loss: {losses[-1]}")

    # getting one hot encoded predictions
    final_probabilities = network.forward(x_train)
    predicted_labels = torch.argmax(final_probabilities, dim=1)
    one_hot_predictions = F.one_hot(predicted_labels, num_classes=3)
    print(one_hot_predictions)

    # mapping encoding to "good", "moderate", and "bad"
    class_indices = torch.argmax(one_hot_predictions, dim=1)
    label_mapping = {0: "good", 1: "moderate", 2: "bad"}
    predicted_labels = [label_mapping[i.item()] for i in class_indices]

    print(f"Accuracy: {get_accuracy(one_hot_predictions, y_one_hot_train): .2f}%")
    plot_training_loss(losses)


if __name__ == "__main__":
    main()
