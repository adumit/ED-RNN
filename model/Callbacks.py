from keras.callbacks import Callback

class TestCallback(Callback):
    def __init__(self, loader, test_steps):
        self.test_loader = loader
        self.num_test_steps = test_steps

    def on_epoch_end(self, epoch, logs={}):
        accuracy = self.model.evaluate_generator(self.test_loader.validation_generator(), steps=self.num_test_steps)
        print('\nTesting acc: {}\n'.format(accuracy))
