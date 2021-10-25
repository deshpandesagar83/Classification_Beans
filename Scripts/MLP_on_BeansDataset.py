"""
NOTE: Future Work
1) Add model save and load functionality
"""
import pandas as pd
import numpy as np
from tempfile import mkdtemp

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


class MlpOnBeans:
    def __init__(self, num_hidden_layers, max_iter, batch_size, learning_rate):
        # Loading the data set and performing some pre-processing operations
        self.__dataframe = pd.read_excel('../Database/Dry_Bean_Dataset.xlsx', header=None)
        self.__dataframe = self.__dataframe[1:]
        self.__dataframe = shuffle(self.__dataframe, random_state=1)
        self.__dataframe = self.__dataframe.reset_index(drop=True)
        print("The Shape of the Data Set is: ", self.__dataframe.shape)

        self.__labels = self.__dataframe.values[:, -1]
        self.__labels = LabelEncoder().fit_transform(self.__labels)
        self.__samples = self.__dataframe.values[:, :-1]
        self.__samples = np.asarray(self.__samples).astype('float32')

        # Default activision = relu, Default solver=adam
        classifier = MLPClassifier(hidden_layer_sizes=num_hidden_layers, max_iter=max_iter,
                                   batch_size=batch_size, learning_rate='adaptive', learning_rate_init=learning_rate,
                                   shuffle=True, random_state=4)

        # Normalizing the values to improve accuracy
        self.__model = make_pipeline(StandardScaler(), classifier, memory=mkdtemp())

    # Test-Train split using sklearn function and evaluating the model
    def eval_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.__samples, self.__labels, test_size=0.10,
                                                            random_state=4)

        self.__model.fit(X_train, y_train)
        print(f'The average score of the model is: {self.__model.score(X_test, y_test):0.03f}')

    # Using cross-validation to evaluate the model
    def evaluate_validate(self):
        scores = cross_val_score(self.__model, self.__samples, self.__labels, cv=10)
        print("The score of each validation iterations is: ")
        counter = 0
        for x in scores:
            counter += 1
            print(f"Iteration Number {counter}: {x:0.03f}")
        print(f"The average score after cross validation is: {np.average(scores):0.03f}")
        print('-----------------------------------------')

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
        self.__model.fit(sam_train, lab_train)
        label_predict = self.__model.predict(sam_test)
        report = classification_report(lab_test, label_predict)
        model_score = self.__model.score(sam_test, lab_test)
        print(f"The model score is: {model_score:0.3f}")
        print("The Classification Report is: ")
        print(report)
        print('-----------------------------------------')
        return model_score

    # Validation and classification
    def validateANDclassify(self, in_length, out_length):
        # Dividing the whole data set into train-test split according to input/output lengths
        df_test = self.__dataframe[in_length:out_length]
        df_train_1 = self.__dataframe[:in_length]
        df_train_2 = self.__dataframe[out_length:]

        if in_length == 0:
            df_train = df_train_2
        elif out_length == self.__dataframe.shape[0]:
            df_train = df_test

        else:
            df_train = pd.concat([df_train_1, df_train_2])

        sample_tr, label_tr = self.samples_labels_split(df_train)
        sample_ts, label_ts = self.samples_labels_split(df_test)
        score = self.classify(sample_tr, sample_ts, label_tr, label_ts)
        return score

    # My own cross-validation fucntion divides into 11 splits, outputs more info
    def my_cross_validate_evaluate(self):
        # Dividing the whole data set into 10 train-test splits to validate them
        max_length = self.__dataframe.shape[0]
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
        print(f'The average score of the model is: {average_score / 11:0.3f}')


"""
NOTE: 
The best values obtained are:
Number of Hidden Layers: 50
Maximum Number of Iterations: 1000
Batch Size: 32
Learning Rate: 0.001
"""

if __name__ == '__main__':
    mlpbeans_object = MlpOnBeans(50, 1000, 32, 0.001)
    mlpbeans_object.eval_train_test()
    mlpbeans_object.evaluate_validate()
    mlpbeans_object.my_cross_validate_evaluate()
