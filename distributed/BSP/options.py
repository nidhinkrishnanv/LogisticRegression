


class Options():
    def __init__(self):

        # Train constants
        self.EPOCH = 12
        # self.num_labels = 49

        self.BATCH_SIZE=200

        
        # Avaiable options : verysmall full
        self.DATA_SIZE = 'verysmall'

        self.lr =1e-3
        # learning rate decay
        self.lr_decay = 2

        self.min_word_count = 4 # Used for removing words with count less than this number.
        self.DEBUG = False