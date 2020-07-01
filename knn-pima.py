import pandas as pd  # for importing and displaying data
import matplotlib.pyplot as plt  # for plotting images
import seaborn as sn  # for displaying confusion matrix
from time import time  # for timing classifications
from scipy import stats  # for collecting the mode of nearest neighbors
from scipy.spatial.distance import pdist  # for calculating distance quickly

data_dir = 'PIMA_data/'
dataset = pd.read_csv(data_dir + 'diabetes.csv')

train = dataset.sample(frac=0.8)
test = dataset.drop(train.index)

# translate datasets into numpy arrays and separate pixel data from labels
X_train = train.drop(['Outcome'], axis=1).values.astype('float32') / 255  # pixels
y_train = train['Outcome'].values.astype('int32')  # labels

X_test = test.drop(['Outcome'], axis=1).values.astype('float32') / 255  # pixels
y_test = test['Outcome'].values.astype('int32')  # labels

distance_metric = ['euclidean', 'minkowski', 'cityblock']

classes = [1, 0]
samples = 30


# find neighbors
def find_neighbors(data_to_label, metric):
    distances = []
    for train_row, label in zip(X_train, y_train):
        distances.append((label, pdist([train_row, data_to_label], metric=metric)[0]))

    distances.sort(key=lambda tup: tup[1])  # Sort list of distances and labels
    return [x[0] for x in distances[0:samples]]  # Return first max_k labels


# predict labels for given data
def predict_label(data, metric):
    p_columns = ['Time Elapsed', '%s Prediction' % metric]
    predictions = pd.DataFrame(columns=p_columns)
    for row in data:
        start_time = time()
        pred = stats.mode(find_neighbors(row, metric))[0][0]
        end_time = time()
        tmp_df = pd.DataFrame([[(end_time - start_time)] + [pred]], columns=p_columns)
        print(tmp_df)
        predictions = predictions.append(tmp_df, ignore_index=True)

    predictions.to_csv(data_dir + 'predictions_' + metric + '.csv')
    return predictions


# check label assigned by k-25
def generate_confusion_matrix(data, labels, name):
    y_actu = pd.Series(labels, name='Actual')
    y_pred = pd.Series(data['%s Prediction' % name].to_list(), name='Predicted')
    confusion_matrix = pd.crosstab(y_actu, y_pred)

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix):
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()
    return None


test_prediction_list = []
test_metrics_list = []
for index, d_metric in enumerate(distance_metric):
    test_prediction_list.append(predict_label(X_test, d_metric))
    test_metrics_list.append(generate_confusion_matrix(test_prediction_list[index], y_test, d_metric))
    plot_confusion_matrix(test_metrics_list[index])
