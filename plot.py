import matplotlib.pyplot as plt

# plot 2 sau mai multe linii pe acelasi grafic
# scatter plot 2d
# scatter plot 3d
# plot learning/loss curves


def learning_loss(history):
    # Plot learning/loss curves from keras fit result
    plt.figure(figsize=(20, 8))
    plt.plot(history.epoch, history.history['loss'], )
    plt.plot(history.epoch, history.history['val_loss'])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title(' Model Loss')
    plt.legend(['Training', 'Dev'])
    plt.show()


def actual_vs_predicted(actual, predicted):
    plt.figure(figsize=(20, 8))
    plt.plot(actual)
    plt.plot(predicted)
    plt.legend(['Actual', 'Predicted'])
    plt.show()
