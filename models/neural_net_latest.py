"""Neural network model."""

from typing import Sequence
from matplotlib.pyplot import axis

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.
    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,

    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.t = 0

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return X @ W + b

    # my fx 
    def linear_grad_W(self, Upstream: np.ndarray, X: np.ndarray,W: np.ndarray ,reg: float = 0.0, m: int = 1 ) -> np.ndarray:
        """The gradient of the error of linear layer w.r.t. W
        Parameters:
            Upstream: the Upstream error
            X: the X data (or input) of the linaer layer
            W: the wieght matrix of linear layer
        Returns:
            the output
        """
        w = W.copy()
        x = X.copy()
        u = Upstream.copy()
        return x.T @ u + reg*w/m

    # my fx
    def linear_grad_X(self, Upstream: np.ndarray, W: np.ndarray, reg: float = 0.0) -> np.ndarray:
        """The gradient of the error of linear layer w.r.t. X (it's input)
        Parameters:
            Upstream: the Upstream error
            W: the Weight Matrix of the linaer layer
        Returns:
            the output
        """
        u = Upstream.copy()
        w = W.copy()
        return u @ w.T
    
    # my fx
    def linear_grad_b(self, Upstream: np.ndarray, b: np.ndarray ,reg: float = 0.0, m: int = 1) -> np.ndarray:
        """The gradient of the error of linear layer w.r.t. b (the bias)
        Parameters:
            Upstream: the Upstream error
        Returns:
            the output
        """
        b_ = b.copy()
        u = Upstream.copy()
        return Upstream.sum(axis = 0) + reg*b_/m


    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me

        return np.where( X < 0, 0, X)

    def relu_grad(self, X: np.ndarray, reg: float = 0.0) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return np.where( X <= 0, 0, 1)
    

    def softmax(self, X: np.ndarray, reg: float = 0.0) -> np.ndarray:
        """The softmax function.
        Parameters:
            X: Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        Returns:
            the output
        """
        # TODO: implement me


        x = X.copy()
        x = x - x.max(axis=1, keepdims=True)
        #adj_x = x - x.max()
        return np.exp(x)/np.sum(np.exp(x), axis = 1).reshape(x.shape[0], 1)
    

    # my fx
    def softmax_grad(self, X: np.ndarray,  y: np.ndarray, reg: float = 0.0) -> np.ndarray:
        """The gradient of softmax function with cross entropy loss.
        Parameters:
            X: Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the softmax funciton after the last layer of your network
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
        Returns:
            the gradient of the softmax with a cross entropy loss 
        """
        # TODO: implement me
        grad = X.copy()
        m = y.shape[0]
        grad[range(m), y] -= 1
        return grad/m


    def calc_cross_entroy_loss(self, Soft_max: np.ndarray,  y: np.ndarray, reg: float = 0.0) -> float:
        """The gradient of softmax function with cross entropy loss.
        Parameters:
            X: Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the softmax funciton after the last layer of your network
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
        Returns:
            the gradient of the softmax with a cross entropy loss 
        """
        m = y.shape[0]

        return np.sum(-np.log(Soft_max[range(m), y])) / m
    
    # my fx
    def calc_l2_reg_loss(self, W: np.ndarray, reg: float = 0.0, m: int = 1) -> float:
        # I'm not putting the "/(2m)" here it's in the 
        return (reg/(2*m)) * np.sum(np.square(W))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.
        
        # the store X as an "output" to be used in Linear 1, R0 for convenience and consistent nomenclature
        self.outputs["R0"] = X

        # for each layer but the last:
        for i in range(1, self.num_layers):
            # Make Linear out, L_i. I didn't end up using this other than as an intermediary before the relu  
            self.outputs["L" + str(i)] = self.linear(
                self.params["W" + str(i)], # The weights of the linear layer
                self.outputs["R" + str(i-1)], # Relu of previous layer, or simply X for 1st Layer
                self.params["b" + str(i)]) # The biases of the linear layer
            # Make Relu outs, R_i
            self.outputs["R" + str(i)] = self.relu(self.outputs["L" + str(i)])
            #print(i)
        
        # make linear out of final layer 
        self.outputs["L" + str(self.num_layers)] = self.linear(
                    self.params["W" + str(self.num_layers)],
                    self.outputs["R" + str(self.num_layers-1)],
                    self.params["b" + str(self.num_layers)]
                    )
        # The Softmax out of the final layer
        self.outputs["S" + str(self.num_layers)] = self.softmax(self.outputs["L" + str(self.num_layers)])
        return self.outputs["S" + str(self.num_layers)]
        

    def backward(self, y: np.ndarray, reg: float = 0.0) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Note: both gradients and loss should include regularization.
        Parameters:
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            reg: Regularization strength
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        

        # initialize loss to zero. Then add CE loss. 
        loss = float(0.0)
        loss += self.calc_cross_entroy_loss(self.outputs["S" + str(self.num_layers)], y)

        # add loss from l2 reg of final layer
        m = len(y) # num of training examples in batch used in L2 loss
        loss+= self.calc_l2_reg_loss(self.params["W" + str(self.num_layers)], reg, m)
        loss+= self.calc_l2_reg_loss(self.params["b" + str(self.num_layers)], reg, m)

        # gradient of Final Softmax
        self.gradients["S" + str(self.num_layers)] = self.softmax_grad(self.outputs["S" + str(self.num_layers)], y )

        # gradients of final linear Layer 
        self.gradients["W" + str(self.num_layers)] = self.linear_grad_W(
            self.gradients["S" + str(self.num_layers)], # Upstream Error
            self.outputs["R" + str(self.num_layers-1)], # the input of the linear layer. (Relu of previous)
            self.params["W" + str(self.num_layers)], # current weights of layer (used in l2 reg grad)
            reg,
            m
        )
        
        #print("# elem after W2: " + str(len(self.gradients)))
        self.gradients["b" +str(self.num_layers)] = self.linear_grad_b(
            self.gradients["S" + str(self.num_layers)], # Upstream Error
            self.params["b" + str(self.num_layers)], # current bias of layer (used in l2 reg grad)
            reg, # reg const
            m # num samples in batch
        )
    

        self.gradients["X" + str(self.num_layers)] = self.linear_grad_X( # gradient of input to lin. layer    
            self.gradients["S" + str(self.num_layers)], # Upstream Error
            self.params["W" + str(self.num_layers)], # Weights of associacted layer
            reg
        )
        # print("X" + str(self.num_layers))
        # print(self.gradients["X" + str(self.num_layers)].shape)
        # print("R" + str(self.num_layers-1))
        # print(self.relu_grad(self.outputs["R" + str(self.num_layers-1)]).shape)

        # gradients of earlier layers. (num layers - i) denotes counting backward
        for i in range(1, self.num_layers):
            # Calc the gradients (W, X, b) for each layer
            self.gradients["W" + str(self.num_layers - i)] = self.linear_grad_W(
                self.gradients["X" + str(self.num_layers - i + 1)]*self.relu_grad(self.outputs["R" + str(self.num_layers-i)]), # Upstream Error
                self.outputs["R" + str(self.num_layers-i-1)], # the input of the linear layer. (Relu of previous)
                self.params["W" + str(self.num_layers -i)], # current weights of layer (used in l2 reg grad)
                reg, # reg const
                m # num samples in batch
            )

            self.gradients["b" +str(self.num_layers - i)] = self.linear_grad_b(
                self.gradients["X" + str(self.num_layers - i + 1)]*self.relu_grad(self.outputs["R" + str(self.num_layers-i)]), # Upstream Error
                self.params["b" + str(self.num_layers - i)], # current bias of layer (used in l2 reg grad)
                reg, # reg const
                m # num samples in batch
            )

            self.gradients["X" + str(self.num_layers - i)] = self.linear_grad_X(
                self.gradients["X" + str(self.num_layers - i + 1)]*self.relu_grad(self.outputs["R" + str(self.num_layers-i)]), # Upstream Error
                self.params["W" + str(self.num_layers - i)], # Weights of lin layer
                reg
            )

            # Add the loss from l2 reg for each layer                            
            loss+= self.calc_l2_reg_loss(self.params["W" + str(self.num_layers - i)], reg, m)
            loss+= self.calc_l2_reg_loss(self.params["b" + str(self.num_layers - i)], reg, m)
    
        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.

        if opt == "SGD": # SGD optimizer
            for i in range(1, self.num_layers + 1):
                self.params["W" + str(i)] -= lr*self.gradients["W" + str(i)]
                self.params["b" + str(i)] -= lr*self.gradients["b" + str(i)]
        
        else: # Implement Adam
            self.t += 1 # what step are we're about to take. I added self.t:int = 0 to constructor

            # if we're taking 1st step
            if self.t == 1:    
                self.m = {} # create m and v dictionaries
                self.v = {}
                # initialize m1 and v1 and update weights
                for i in range(1, self.num_layers + 1): # for each layer
                    self.m["W" + str(i) ] = (1-b1) * self.gradients["W" + str(i)] # calc initial m for Wi
                    self.m["b" + str(i) ] = (1-b1) * self.gradients["b" + str(i)] # calc initial m for bi
                    self.v["W" + str(i) ] = (1-b2) * np.square(self.gradients["W" + str(i)]) # calc initial v for Wi
                    self.v["b" + str(i) ] = (1-b2) * np.square(self.gradients["b" + str(i)]) # calc initial v for bi
                    
                    # Then update weights
                    m_hat_w = self.m["W" + str(i) ] / (1-b1)
                    v_hat_w = self.v["W" + str(i)] / (1-b2)
                    self.params["W" + str(i)] -= lr*m_hat_w/(np.sqrt(v_hat_w) + eps)

                    m_hat_b = self.m["b" +str(i)] / (1-b1**self.t)
                    v_hat_b = self.v["b" +str(i)] / (1-b2**self.t)
                    self.params["b" + str(i)] -= lr*m_hat_b/(np.sqrt(v_hat_b) + eps)

            # for steps 2+        
            else:
                for i in range(1, self.num_layers + 1):
                    self.m["W" + str(i) ] = (b1)*self.m["W" + str(i) ] + (1-b1) * self.gradients["W" + str(i)] # update mt for Wi
                    self.m["b" + str(i) ] = (b1)*self.m["b" + str(i) ] + (1-b1) * self.gradients["b" + str(i)] # update mt for bi
                    self.v["W" + str(i) ] = (b2)*self.v["W" + str(i) ] + (1-b2) * np.square(self.gradients["W" + str(i)]) # update vt for Wi
                    self.v["b" + str(i) ] = (b2)*self.v["b" + str(i) ] + (1-b2) * np.square(self.gradients["b" + str(i)]) # update vt for bi 
                
                    # Then update weights
                    m_hat_w = self.m["W" + str(i) ] / (1-b1**self.t)
                    v_hat_w = self.v["W" + str(i)] / (1-b2**self.t)

                    self.params["W" + str(i)] -= lr*m_hat_w/(np.sqrt(v_hat_w) + eps)

                    m_hat_b = self.m["b" +str(i)] / (1-b1**self.t)
                    v_hat_b = self.v["b" +str(i)] / (1-b2**self.t)

                    self.params["b" + str(i)] -= lr*m_hat_b/(np.sqrt(v_hat_b) + eps)



                    