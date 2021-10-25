import pandas as pd
import numpy as np
from tempfile import mkdtemp

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


class SVMonBeans:
    def __init__(self):
        self.__dataframe = pd.read_excel('../Database/Dry_Bean_Dataset.xlsx', header=None)
        self.__dataframe = self.__dataframe[1:]
        self.__dataframe = shuffle(self.__dataframe, random_state=1)
        self.__dataframe = self.__dataframe.reset_index(drop=True)
        print("The Shape of the Data Set is: ", self.__dataframe.shape)

        self.__labels = self.__dataframe.values[:, -1]
        self.__labels = LabelEncoder().fit_transform(self.__labels)
        self.__samples = self.__dataframe.values[:, :-1]
        self.__samples = np.asarray(self.__samples).astype('float32')

    def eval_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.__samples, self.__labels, test_size=0.10,
                                                            random_state=4)

        clf = make_pipeline(StandardScaler(),
                            SVC(kernel='linear', cache_size=1000, gamma='auto', decision_function_shape='ovo',
                                random_state=1),
                            memory=mkdtemp())

        clf.fit(X_train, y_train)
        print(f'The average score of the model is: {clf.score(X_test, y_test):0.03f}')

    def evaluate_validate(self):
        model = make_pipeline(StandardScaler(), SVC(kernel='linear', cache_size=1000, gamma='auto',
                                                    decision_function_shape='ovo', random_state=1),
                              memory=mkdtemp())
        # model = SVC(kernel='poly', C=1, random_state=42, gamma='auto')

        scores = cross_val_score(model, self.__samples, self.__labels, cv=10)
        print("The score of each validation iterations is: ")
        counter = 0
        for x in scores:
            counter += 1
            print(f"Iteration Number {counter}: {x:0.03f}")
        print(f"The average score after cross validation is: {np.average(scores):0.03f}")


if __name__ == '__main__':
    SVM_Example = SVMonBeans()
    SVM_Example.eval_train_test()
    SVM_Example.evaluate_validate()
