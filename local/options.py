


class Options():
    def __init__(self):

        # Train constants
        self.EPOCH = 30
        # self.num_labels = 49
        
        # Avaiable options : verysmall full
        self.DATA_SIZE = 'full'

        # learning rate decat
        self.lr_decay = 4

        self.min_word_count = 3 # Used for removing words with count less than this number.
        self.DEBUG = False