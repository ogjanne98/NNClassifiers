import NNClassifier

class MultiClass(NNClassifier):
    """Multiclass classification"""
    def __init__(self):
        super().__init__()


class ReLUSoftCCE(MultiClass):
    """
    Hidden activatiion: ReLU
    Output activation: Softmax
    Loss: CCE
    """
    def __init__(self):
        super().__init__()        