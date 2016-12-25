import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from cs231n.classifiers.fc_net import *
import matplotlib.pyplot as plt
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

weight_scale = 3.862881e-02
learning_rate = 1.946705e-03


print "loading train vector"
train = np.loadtxt(r'C:\Users\roeyvi\Dropbox\University\Year_4\ML\Project\train_tfidf.csv', delimiter =',')
#print "loading test vector"
#test = np.loadtxt(r'C:\Users\roeyvi\Dropbox\University\Year_4\ML\Project\test_tfidf.csv', delimiter =',')

print train.shape
#print test.shape

labels = [0 for i in range(12500)] + [1 for j in range(12500)]

train_range = range(10000) + range(24999, 24999-10000, -1)
val_range = range(10000, 10000+5000, 1)

data = {
    'X_train': train[train_range, :],
    'y_train': labels[train_range, :],
    'X_val': train[val_range, :],
    'y_val': labels[val_range, :]
  }

learning_rates = 10 ** np.random.uniform(-7, 0, 2)
weight_scales = 10 ** np.random.uniform(-4, 0, 2)
results = {}
for lr in learning_rates:
    for ws in weight_scales:
        print "learning rate: %f, weight scales: %f" % (lr, ws)
        model = FullyConnectedNet([100, 100, 100, 100],
                                  weight_scale=ws, dtype=np.float64, use_batchnorm=False, reg=1e-2)
        solver = Solver(model, data,
                        print_every=100, num_epochs=10, batch_size=25,
                        update_rule='adam',
                        optim_config={
                            'learning_rate': lr,
                        },
                        lr_decay=0.9,
                        verbose=True
                        )

        solver.train()
        train_acc = solver.train_acc_history[-1]
        val_acc = solver.val_acc_history[-1]
        results[(lr, ws)] = train_acc, val_acc

        plt.subplot(2, 1, 1)
        plt.plot(solver.loss_history)
        plt.title('Loss history')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.subplot(2, 1, 2)
        plt.plot(solver.train_acc_history, label='train')
        plt.plot(solver.val_acc_history, label='val')
        plt.title('Classification accuracy history')
        plt.xlabel('Epoch')
        plt.ylabel('Classification accuracy')
        plt.show()

plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()

