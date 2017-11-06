import numpy as np

class LogisticReg():

    # Intial weights for logistic regression
    def __init__(self, input_dim, output_dim, lr=0.001, lamda=0.001):
        self.W = np.zeros((output_dim, input_dim))
        self.b = np.zeros((output_dim, 1))
        self.lr = lr
        self.lamda = lamda



    def forward(self, inputs):
        out = np.matmul(self.W, inputs) +self.b
        return self.sigmoid(out)



    def train_step(self, inputs, labels, lr):

        self.lr = lr

        # Get the output vector containing the probability of each class.
        inputs = inputs.reshape(inputs.shape[0], 1)
        labels = labels.reshape(labels.shape[0], 1)
        probs = self.forward(inputs)

        # get the gradients
        dW = np.matmul((probs - labels), inputs.T) + self.lamda * self.W
        db = probs - labels + self.lamda*self.b

        # Update the weight matrix and b with gradients
        self.W = self.W - self.lr*dW
        self.b = self.b - self.lr*db

        # Get the index with probability greater than one.
        preds = probs > 0.5

        # Get exact accuracy 
        acc = self.get_exact_acc(preds, labels)

        return acc

    def test_step(self, inputs, labels):
        # Get the output vector containing the probability of each class.
        inputs = inputs.reshape(inputs.shape[0], 1)
        labels = labels.reshape(labels.shape[0], 1)
        probs = self.forward(inputs)

        # Get the index with probability greater than one.
        preds = probs > 0.5

        # Get exact accuracy 
        acc = self.get_exact_acc(preds, labels)

        return acc



       

    def get_exact_acc(self, preds, labels):
         # Get exact match
        matches = np.sum(preds == labels) # labels is a vector with true for correct label.
        return matches/len(labels)

    def precision(self, preds, labels):
        true_pos = np.sum(np.logical_and((preds==labels), preds))
        return true_pos/(np.sum(preds))

    def recall(self, preds, labesl):
        true_pos = np.sum(np.logical_and(preds==labels, preds))
        return true_pos/np.sum(labels)

    def f1_score(preds, labels):
        prec = self.precision(preds, labels)
        recall = self.recall(preds, labels)
        return 2*(pred * recall)/(pred + recall)


    # Helper functions
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))


if __name__ == '__main__':
    lg = LogisticReg(5, 2)
    out = lg.output(np.array([4, 5, 6, 1, 2]))
    print(out)


