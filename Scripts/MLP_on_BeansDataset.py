"""
NOTE: Future Work
1) Improve the score of the model
2) Use more than just score to determine the accuracy of the model
3) Come up with a method other than brute force to determine the input metrics of the classifier
4) Add model save and load functionality
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle


class MlpOnBeans:
    def __init__(self, num_hidden_layers, max_iter, batch_size, learning_rate):
        self.dataframe = []
        # Default activision = relu, Default solver=adam
        self.model = MLPClassifier(hidden_layer_sizes=num_hidden_layers, max_iter=max_iter,
                                   batch_size=batch_size, learning_rate='adaptive', learning_rate_init=learning_rate,
                                   shuffle=True, random_state=4)

    # Loading the data set and performing some pre-processing operations
    def load_data_set(self):
        self.dataframe = pd.read_excel('../Database/Dry_Bean_Dataset.xlsx', header=None)
        self.dataframe = self.dataframe[1:]
        self.dataframe = shuffle(self.dataframe, random_state=1)
        self.dataframe = self.dataframe.reset_index(drop=True)
        print("The Shape of the Data Set is: ", self.dataframe.shape)
        return self.dataframe.shape[0]

    # Dividing the whole data set into train-test split according to input/output lengths
    def train_test_split(self, in_length, out_length):
        df_test = self.dataframe[in_length:out_length]
        df_train_1 = self.dataframe[:in_length]
        df_train_2 = self.dataframe[out_length:]

        if in_length == 0:
            return df_train_2, df_test
        if out_length == self.dataframe.shape[0]:
            return df_train_1, df_test

        df_train = pd.concat([df_train_1, df_train_2])
        return df_train, df_test

    # Splitting the labels and samples and some more pre-processing
    @staticmethod
    def samples_labels_split(input_dataframe):
        labels = input_dataframe.values[:, -1]
        labels = LabelEncoder().fit_transform(labels)
        samples = input_dataframe.values[:, :-1]
        samples = np.asarray(samples).astype('float32')
        return samples, labels

    # Main classifier logic
    def classify(self, sam_train, sam_test, lab_train, lab_test):
        self.model.fit(sam_train, lab_train)
        model_score = self.model.score(sam_test, lab_test)
        print(f"The model score is: {model_score:0.3f}")
        return model_score

    # Validation and classification
    def validateANDclassify(self, test_val_beg, test_val_end):
        df_train, df_test = self.train_test_split(test_val_beg, test_val_end)
        sample_tr, label_tr = self.samples_labels_split(df_train)
        sample_ts, label_ts = self.samples_labels_split(df_test)
        score = self.classify(sample_tr, sample_ts, label_tr, label_ts)
        return score


"""
NOTE: Further optimization on these values is Required
By using different combinations the maximum score can be attained by using these values for the variables
Number of Hidden Layers: 50
Maximum Number of Iterations: 1000
Batch Size: 32
Learning Rate: 0.001
"""

if __name__ == '__main__':
    mlpbeans_object = MlpOnBeans(50, 1000, 32, 0.001)
    max_length = mlpbeans_object.load_data_set()

    # Dividing the whole data set into 10 train-test splits to validate them
    average_score = 0
    test_val_beg = 0
    increment_val = max_length / 10
    increment_val = int(np.floor(increment_val / 100.0) * 100)
    test_val_end = increment_val

    # Since the data set has 13611 samples it will be divided into 11 validation splits
    for x in range(11):
        print(f'Iteration Number:{x}\t\tTesting Set Index: {test_val_beg}-{test_val_end}')
        average_score += mlpbeans_object.validateANDclassify(test_val_beg, test_val_end)
        if x == 9:
            test_val_end = max_length
        else:
            test_val_end += increment_val
        test_val_beg += increment_val

    print('-----------------------------------------')
    print(f'The average score of the model is: {average_score/11:0.3f}')
