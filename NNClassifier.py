import numpy as np

class NNClassifier:
    def __init__(self):
        pass

    def fit(self, X, y, X_val, y_val, hidden_layers=[1], batch_size=None, max_epochs=100, lr=0.1):
        """
        Description:
        Fit the neural network model to the data using gradient descent and backpropagation.

        Parameters:
        X (array): The training data. NxM - N samples, M features.
        y (array): The training targets. NxK - N samples, K classes.
        X_val (array): The validation data.
        y_val (array): The validation targets.
        hidden_layers (list): List with number of nodes in each layer, starting from the leftmost hidden layer.
        batch_size (int): Size of batch. Between 1 and N.
        max_epochs (int): Maximum number of epochs before stoppage.
        lr (float): The learning rate.

        Returns:
        
        """
        # Initialize properties
        self.n_features = X.shape[1]
        self.n_classes = y.shape[1] 
        self.weights = self.initialize_weights([self.n_features] + hidden_layers + [self.n_classes]) # Array of weight matrices
        self.X_val = X_val
        self.y_val = y_val
        self.n_val_loss_increase = 0
        self.val_loss = 0
        if batch_size != None:
            self.batch_size = batch_size
        else:
            self.batch_size = X.shape[0]

        # Main loop
        self.stopping_condition = False 
        epoch = 0
        while (not self.stopping_condition) and (epoch < max_epochs):
            self.update_weights()
            self.update_condition()
            epoch += 1


    def predict(self, X):
        """
        Description:
        Predict output from data.

        Parameters: 
        X (array): The data.

        Returns:
        (array): The predicted output.
        """
        return np.argmax(self.forward(X), axis=1) 

    def initialize_weigths(self, layers):
        """
        Description:
        Initialize the weight matrices in the neural network uniform random numbers in [-1,1].

        Parameters:
        layers (list): The number of nodes in each layer, from the input to the output.

        Returns:
        weights (list): All of the weights matrices, from the first to the last.
        """
        weights = []
        for l1,l2 in zip(layers[:-1], layers[1:]):
            weights.append(np.random.uniform(low=-1, high=1, size=(l1,l2)))
        return weights    

    def update_condition(self):
        """
        Description:
        Updates early stoppage condition in neural network. Stops if the validation loss increases over several epochs in a row.
        Stops after strictly greater than 3 epochs.
        """
        ls = self.loss(self.X_val, self.y_val)
        if  ls > self.val_loss:
            self.n_val_loss_increase += 1
        else:
            self.n_val_loss_increase = 0
        self.val_loss = ls  
        self.stopping_condition = (self.n_val_loss_increase > 3)      

    def update_weights(self):
        raise NotImplementedError("Superclass not to be instantiated")  
    
    def forward(self):
        raise NotImplementedError("Superclass not to be instantiated")
    
    def loss(self):
        raise NotImplementedError("Superclass not to be instantiated")
