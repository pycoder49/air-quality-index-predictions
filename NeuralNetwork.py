from typing import List, Tuple
import torch


class NeuralNetwork:
    def __init__(self,
                 *hidden_layers: int,
                 input_size: int,
                 output_size: int,
                 learning_rate: float):
        # ---------- CUSTOMIZABLE FEATURES ---------- #
        self.output_size = output_size
        self.layers = [input_size] + list(hidden_layers) + [output_size]
        self.hidden_layer_count = len(self.layers)

        # ---------- NEURAL NET FEATURES ---------- #
        self.Z: List[torch.Tensor] = []  # linear transformations
        self.A: List[torch.Tensor] = []  # activation outputs
        self.pred: torch.Tensor = torch.zeros(0)  # predictions from forward pass
        self.grads: List[Tuple[torch.Tensor, torch.Tensor]] = []  # gradients for back prop
        self.w: List[torch.Tensor]  # list of weights
        self.b: List[torch.Tensor]  # list of biases

        self.w = [
            torch.rand(self.layers[i + 1], self.layers[i]) * 0.01
            for i in range(len(self.layers) - 1)
        ]
        self.b = [
            torch.zeros(self.layers[i + 1])
            for i in range(len(self.layers) - 1)
        ]

        # ---------- ADAM OPTIMIZER PARAMETERS ---------- #
        self.m_w: List[torch.Tensor] = []  # momentum for weights
        self.v_w: List[torch.Tensor] = []  # velocity for weights
        self.m_b: List[torch.Tensor] = []  # momentum for biases
        self.v_b: List[torch.Tensor] = []  # velocity for biases

        self.learning = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8  # prevent division by 0
        self.t = 0  # timestep

        # ---------- INITIALIZING MOMENTUM AND VELOCITIES ---------- #
        num_layers = len(self.w)  # Ensure we match self.w and self.b correctly

        for i in range(num_layers):
            self.m_w.append(torch.zeros_like(self.w[i]))
            self.v_w.append(torch.zeros_like(self.w[i]))
            self.m_b.append(torch.zeros_like(self.b[i]))
            self.v_b.append(torch.zeros_like(self.b[i]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagates through all layers, giving you predictions for input data using current weights and bias
        :param x: training data
        :return: predictions using current weights and bias
        """
        # clearing out lists to avoid stacking
        self.Z.clear()
        self.A.clear()

        # going through the network
        activation = x
        for i in range(len(self.layers) - 1):
            logit = torch.matmul(activation, self.w[i].T) + self.b[i]
            activation = self._relu(logit) if i != len(self.layers) - 2 else self._softmax(logit)
            self.Z.append(logit)
            self.A.append(activation)

        # return softmax output probabilities
        return self.A[-1]

    def backprop(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Equations used for calculations:
        @ TODO: explain equations used for clarification

        :param x: input training data tensor
        :param y: one hot encoded tensor
        :return: None
        """
        # clearing out gradients to avoid stacking
        self.grads.clear()

        start = len(self.layers) - 1  # starting from output layer
        delta = self.A[-1] - y
        for i in range(start, 0, -1):
            # if i == 1, then we're calculating gradients for first HL, and we need the raw input for that, and
            # not the activation of the input
            tensor = self.A[i - 2] if i != 1 else x
            dl_dw = torch.matmul(tensor.T, delta)
            dl_db = torch.sum(delta, dim=0, keepdim=True)
            self.grads.append((dl_dw, dl_db))

            if i != 1:      # delta for the next iteration
                delta = torch.matmul(delta, self.w[i - 1]) * self._relu_derivative(self.Z[i - 2])

    def update_parameters(self, adam_step: bool = False) -> None:
        # extracting weights and bias
        dw = [grad[0] for grad in self.grads]       # [dw3, dw2, dw1...]
        db = [grad[1] for grad in self.grads]       # [db3, db2, db1...]

        # reversing the list so that the list is ascending
        dw = dw[::-1]
        db = db[::-1]

        if adam_step:
            # Increment time step
            self.t += 1

            # updating part
            for i in range(len(self.w)):
                # updating momentum
                self.m_w[i] = self.beta1*self.m_w[i] + (1-self.beta1)*dw[i].T
                self.m_b[i] = self.beta1*self.m_b[i] + (1-self.beta1)*db[i]

                # update velocity
                self.v_w[i] = self.beta2*self.v_w[i] + (1-self.beta2)*(dw[i].T**2)
                self.v_b[i] = self.beta2*self.v_b[i] + (1-self.beta2)*(db[i]**2)

                # computing bias-corrected moments and velocities
                m_w_corrected = self.m_w[i] / (1 - self.beta1 ** self.t)
                m_b_corrected = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_w_corrected = self.v_w[i] / (1 - self.beta2 ** self.t)
                v_b_corrected = self.v_b[i] / (1 - self.beta2 ** self.t)

                # updating the parameters
                self.w[i] = self.w[i] - self.learning * m_w_corrected / (torch.sqrt(v_w_corrected) + self.epsilon)
                self.b[i] = self.b[i] - self.learning * m_b_corrected / (torch.sqrt(v_b_corrected) + self.epsilon)
        else:
            # normal updates without momentum and velocities
            for i in range(len(self.w)):
                self.w[i] = self.w[i] - self.learning * dw[i].T
                self.b[i] = self.b[i] - self.learning * db[i]

    def calculate_loss(self, predictions: torch.tensor, y: torch.Tensor) -> float:
        """
        Cross Entropy Loss

        :param predictions: predictions tensor containing probabilities
        :param y: target predictions
        :return: loss of current predictions
        """
        return -torch.mean(torch.sum(y * torch.log(predictions + 1e-8), dim=1))  # mean loss

    def _relu(self, z: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.tensor(0.0), z)

    def _relu_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float()

    def _softmax(self, z: torch.Tensor) -> torch.Tensor:
        exp_z = torch.exp(z - torch.max(z, dim=1, keepdim=True).values)
        return exp_z / torch.sum(exp_z, dim=1, keepdim=True)  # sum over classes
