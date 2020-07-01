import pandas as pd  # for importing and displaying data
import matplotlib.pyplot as plt  # for plotting images
import seaborn as sn  # for displaying confusion matrix
from time import time  # for timing classifications
from scipy import stats  # for collecting the mode of nearest neighbors
from scipy.spatial.distance import pdist  # for calculating distance quickly

data_dir = 'MNIST_data/'
dataset = pd.read_csv(data_dir + 'train.csv')
train = dataset.sample(frac=0.8)
test = dataset.drop(train.index)

# translate datasets into numpy arrays and separate pixel data from labels
X_train = train.drop(['label'], axis=1).values.astype('float32') / 255  # pixels
y_train = train['label'].values.astype('int32')  # labels

X_test = test.drop(['label'], axis=1).values.astype('float32') / 255  # pixels
y_test = test['label'].values.astype('int32')  # labels

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']  # used for display
distance_metric = "euclidean"  # distance metric used
samples = 100  # maximum of k
p_columns = ['Time Elapsed'] + ['Pred k %d' % k for k in range(1, samples + 1)]  # columns for dataframe


# find neighbors
def find_neighbors(max_k, data_to_label):
    distances = []
    for train_row, label in zip(X_train, y_train):
        distances.append((label, pdist([train_row, data_to_label], metric=distance_metric)[0]))

    distances.sort(key=lambda tup: tup[1])  # Sort list of distances and labels
    return [x[0] for x in distances[0:max_k]]  # Return first max_k labels


# predict labels for given data
def predict_label(data, name):
    predictions = pd.DataFrame(columns=p_columns)
    index = 0
    for row in data:
        start_time = time()
        pred = find_neighbors(samples, row)
        end_time = time()
        pred_by_k = [stats.mode(pred[0:k])[0][0] for k in range(1, samples + 1)]
        tmp_df = pd.DataFrame([[(end_time - start_time)] + pred_by_k], columns=p_columns)
        print({'Index': index, 'Time Elapsed': (end_time - start_time)})
        predictions = predictions.append(tmp_df, ignore_index=True)
        index += 1

    print(predictions)
    file_n = 'predictions_' + name + '.csv'
    predictions.to_csv(data_dir + file_n)
    return predictions


# check label assigned by k-25
def generate_confusion_matrix(data, labels, k):
    y_actu = pd.Series(labels, name='Actual')
    y_pred = pd.Series(data[p_columns[k+1]].to_list(), name='Predicted')
    confusion_matrix = pd.crosstab(y_actu, y_pred)

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix):
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()
    return None


# get test metrics
test_predictions = predict_label(X_test, 'test')
test_metrics = generate_confusion_matrix(test_predictions, y_test, 30)
plot_confusion_matrix(test_metrics)
