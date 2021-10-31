# Import Pandas, Numpy, Scipy, and Matplotlib Python Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# -------------------------------------------------------------
# Data Preprocessing Step
# -------------------------------------------------------------

labels = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
mushrooms = pd.read_csv('./data/mushrooms.csv')

# Clean the data
mushrooms_clean = mushrooms.copy()
mushrooms_clean['class'] = mushrooms_clean['class'].replace({'e': 'Edible', 'p': 'Poisonous'})
mushrooms_clean[labels[0]] = mushrooms_clean[labels[0]].replace({'b': 'Bell', 'c': 'Conical', 'x': 'Convex', 'f': 'Flat', 'k': 'Knobbed', 's': 'Sunken'})
mushrooms_clean[labels[1]] = mushrooms_clean[labels[1]].replace({'f': 'Fibrous', 'g': 'Grooves', 'y': 'Scaly', 's': 'Smooth'})
mushrooms_clean[labels[2]] = mushrooms_clean[labels[2]].replace({'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 'r': 'Green', 'p': 'Pink', 'u': 'Purple', 'e': 'Red', 'w': 'White', 'y': 'Yellow'})
mushrooms_clean[labels[3]] = mushrooms_clean[labels[3]].replace({'t': 'Bruises', 'f': 'No'})
mushrooms_clean[labels[4]] = mushrooms_clean[labels[4]].replace({'a': 'Almond', 'l': 'Anise', 'c': 'Creosote', 'y': 'Fishy', 'f': 'Foul', 'm': 'Musty', 'n': 'None', 'p': 'Pungent', 's': 'Spicy'})
mushrooms_clean[labels[5]] = mushrooms_clean[labels[5]].replace({'a': 'Attached', 'd': 'Descending', 'f': 'Free', 'n': 'Notched'})
mushrooms_clean[labels[6]] = mushrooms_clean[labels[6]].replace({'c': 'Close', 'w': 'Crowded', 'd': 'Distant'})
mushrooms_clean[labels[7]] = mushrooms_clean[labels[7]].replace({'b': 'Broad', 'n': 'Narrow'})
mushrooms_clean[labels[8]] = mushrooms_clean[labels[8]].replace({'k': 'Black', 'n': 'Brown', 'b': 'Buff', 'h': 'Chocolate', 'g': 'Grey', 'r': 'Green', 'o': 'Orange', 'p': 'Pink', 'u': 'Purple', 'e': 'Red', 'w': 'White', 'y': 'Yellow'})
mushrooms_clean[labels[9]] = mushrooms_clean[labels[9]].replace({'e': 'Enlarged', 't': 'Tampering'})
mushrooms_clean[labels[10]] = mushrooms_clean[labels[10]].replace({'b': 'Bulbous', 'c': 'Club', 'u': 'Cup', 'e': 'Equal', 'z': 'Rhizomorphs', 'r': 'Rooted', '?': 'Missing'})
mushrooms_clean[labels[11]] = mushrooms_clean[labels[11]].replace({'f': 'Fibrous', 'y': 'Scaly', 'k': 'Silky', 's': 'Smooth'})
mushrooms_clean[labels[12]] = mushrooms_clean[labels[12]].replace({'f': 'Fibrous', 'y': 'Scaly', 'k': 'Silky', 's': 'Smooth'})
mushrooms_clean[labels[13]] = mushrooms_clean[labels[13]].replace({'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 'o': 'Orange', 'p': 'pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow'})
mushrooms_clean[labels[14]] = mushrooms_clean[labels[14]].replace({'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 'o': 'Orange', 'p': 'pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow'})
mushrooms_clean[labels[15]] = mushrooms_clean[labels[15]].replace({'p': 'Partial', 'u': 'Universal'})
mushrooms_clean[labels[16]] = mushrooms_clean[labels[16]].replace({'n': 'Brown', 'o': 'Oranage', 'y': 'Yellow', 'w': 'White'})
mushrooms_clean[labels[17]] = mushrooms_clean[labels[17]].replace({'n': 'None', 'o': 'One', 't': 'Two'})
mushrooms_clean[labels[18]] = mushrooms_clean[labels[18]].replace({'c': 'Cobwebby', 'e': 'Evanescent', 'f': 'Flaring', 'l': 'Large', 'n': 'None', 'p': 'Pendant', 's': 'Sheathing', 'z': 'Zone'})
mushrooms_clean[labels[19]] = mushrooms_clean[labels[19]].replace({'k': 'Black', 'n': 'Brown', 'b': 'Buff', 'h': 'Chocolate', 'r': 'Green', 'o': 'Orange', 'u': 'Purple', 'w': 'White', 'y': 'Yellow'})
mushrooms_clean[labels[20]] = mushrooms_clean[labels[20]].replace({'a': 'Abundant', 'c': 'Clustered', 'n': 'Numerous', 's': 'Scattered', 'v': 'Several', 'y': 'Solitary'})
mushrooms_clean[labels[21]] = mushrooms_clean[labels[21]].replace({'g': 'Grasses', 'l': 'Leaves', 'm': 'Meadows', 'p': 'Paths', 'u': 'Urban', 'w': 'Waste', 'd': 'Woods'})

# Create categorical data
mushrooms_clean = pd.get_dummies(mushrooms_clean, columns=labels, prefix=labels)
print(mushrooms_clean.head())



# -------------------------------------------------------------
# EDA Step
# -------------------------------------------------------------

# Number of edible and non-edible mushrooms
ax = mushrooms['class'].value_counts().plot(kind='bar', rot=0)
ax.set_ylabel('Count')
ax.set_xlabel('Class of Mushroom')
ax.set_title('Number of Edible vs Non-Edible Mushrooms')
plt.show()

# Cap comparision
# ax = plt.figure().add_axes([0, 0, 1, 1])
# ax = mushrooms['cap-color'].value_counts().plot(kind='bar')
# ax.set_ylabel('Count')
# ax.set_xlabel('Cap Color of Mushroom')
# ax.set_title('Types of Cap Color for Mushrooms')
# plt.show()
# ax = mushrooms['cap-shape'].value_counts().plot(kind='bar', rot=0)
# ax.set_ylabel('Count')
# ax.set_xlabel('Cap Shape of Mushroom')
# ax.set_title('Types of Cap Shape for Mushrooms')
# plt.show()
# ax = mushrooms['cap-surface'].value_counts().plot(kind='bar', rot=0)
# ax.set_ylabel('Count')
# ax.set_xlabel('Cap Surface of Mushroom')
# ax.set_title('Types of Cap Surface for Mushrooms')
# plt.show()



# -------------------------------------------------------------
# Decision Tree Algorithm
# -------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

feature_column = mushrooms_clean[['odor_Almond', 'odor_Anise', 'odor_Creosote', 'odor_Fishy', 'odor_Foul', 'odor_Musty', 'odor_None', 'odor_Pungent', 'odor_Spicy', 'habitat_Grasses', 'habitat_Leaves', 'habitat_Meadows', 'habitat_Paths', 'habitat_Urban', 'habitat_Waste', 'habitat_Woods']]
target_column = mushrooms_clean['class']

feature_train, feature_test, target_train, target_test = train_test_split(feature_column, target_column, test_size=0.3, random_state=1)

classifier = DecisionTreeClassifier()
classifier = classifier.fit(feature_train, target_train)
target_pred = classifier.predict(feature_test)

print('Accuracy: ', metrics.accuracy_score(target_test, target_pred))