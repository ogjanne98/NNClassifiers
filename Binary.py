import NNClassifier

class Binary(NNClassifier):
    """Binary classification."""
    def __init__(self):
        super().__init__()


class ReLUSigBCE(Binary):
    """
    Hidden activation: ReLU
    Output activation: Sigmoid
    Loss: BCE
    """
    def __init__(self):
        super().__init__()