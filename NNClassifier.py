
class NNClassifier:
    def __init__(self):
        pass

    def fit(self, X, y, X_val, y_val, hidden_layers=[1], batch_size="full", max_epochs=100, lr=0.1):
        # initialize properties
        self.n_features = X.shape[1]
        self.n_classes = y.shape[1] #assume that targets are 1-hot
        self.weights = self.initialize_weights() #initialize weights
        self.batch_size = batch_size

        stopping_condition = False # to be defined
        epoch = 0
        while (not stopping_condition) and (epoch < max_epochs):
            self.update_weights()
            stopping_condition = self.update_condition()
            epoch += 1


    def predict(self):
        # to be implemented    

    def initialize_weigths(self):
        # to be implemented

    def update_condition(self):
        # to be implemented    

    def update_weights(self):
        raise NotImplementedError("Superclass not to be instantiated")  
    
    def forward(self):
        raise NotImplementedError("Superclass not to be instantiated")
