#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:09:37 2017

Evaluation Metrics
"""

import time
import numpy as np

from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt


class EvaluationMetrics:
    
    def __init__(self, classifier, X, y, classifier_name):
        self.classifier = classifier
        self.X = X
        self.y = y
        self.kfold = model_selection.KFold(n_splits=10)
        self.classifier_name = classifier_name
        
    def cross_validate_for_accuracy(self):
        results = model_selection.cross_val_score(self.classifier, self.X, self.y, cv=self.kfold, scoring='accuracy')
        print("Classifier "+self.classifier_name+" - Accuracy: %.10f (%.10f)") % (results.mean(), results.std())
        
    def cross_validate_auc_roc(self):
        y = label_binarize(self.y, classes = [-1, 0, 1])
        
        n_classes = y.shape[1]
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size = 0.1, random_state = 0)
        
        onvRestClassifier = OneVsRestClassifier(self.classifier)
        y_score = onvRestClassifier.fit(X_train, y_train).decision_function(X_test)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def cross_validate_precision_score(self):
        scorer = make_scorer(precision_score, average="micro")
        results = model_selection.cross_val_score(self.classifier, self.X, self.y, cv=self.kfold, scoring=scorer)
        print("Classifier "+self.classifier_name+" - Precision Score: %.10f (%.10f)") % (results.mean(), results.std())
    
    def cross_validate_confusion_matrix(self):
        
        kf = cross_validation.KFold(len(self.y), n_folds=10)
        
        self.y = np.array(self.y)
        
        for train_index, test_index in kf:
           X_train, X_test = self.X[train_index], self.X[test_index]
           y_train, y_test = self.y[train_index], self.y[test_index]

           self.classifier.fit(X_train, y_train)
           #print ("Classifier "+self.classifier_name + " Confusion Matrix ",confusion_matrix(y_test, self.classifier.predict(X_test)))
           #print (confusion_matrix(y_test, self.classifier.predict(X_test)))
        

    def perform_metrics(self):
        cross_validate_confusion_matrix()
        cross_validate_precision_score()
        cross_validate_auc_roc()
        cross_validate_for_accuracy()
        