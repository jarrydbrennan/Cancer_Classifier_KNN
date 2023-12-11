import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#load data
breast_cancer_data = load_breast_cancer()
#inspect data
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)
##Target: Binary classification of malignant or benign
# print(breast_cancer_data.target)
# print(breast_cancer_data.target_names)

#Splitting data into training & validation sets
training_data,validation_data,training_labels,validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)
#Are the training sets the same size?
print(len(training_data) == len(training_labels))

#Running the Classifier
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(training_data,training_labels)
#How accurate is the validation set?
print(classifier.score(validation_data,validation_labels))

#Better k? Loop above to find best k
accuracies = []
for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data,training_labels)
  accuracies.append(classifier.score(validation_data,validation_labels))

#Graph Results K vs Accuracy
k_list = range(1,101)
plt.plot(k_list,accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()