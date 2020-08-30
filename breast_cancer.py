# Import necessary packages and libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt
breast_cancer_data = load_breast_cancer()
# print data to terminal to view it
print(breast_cancer_data)
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)
# Splitting data for training and testing
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data,breast_cancer_data.target,test_size = 0.2,random_state = 100)
print(len(training_data))
print(len(training_labels))
# finding accuracies for different values of k
accuracies = []
for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data,training_labels)
  accuracies.append(classifier.score(validation_data,validation_labels))
k_list = range(1,101)
plt.plot(k_list,accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
