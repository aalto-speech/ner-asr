import seaborn as sns
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer  


predicted = np.load('digitoday_predict.npy', allow_pickle=True)
true = np.load('digitoday_true.npy', allow_pickle=True)


def flatten(array):
    flattened = []
    for i in array:
        for j in i:
            flattened.append(j)
    return flattened

predicted = flatten(predicted)
true = flatten(true)

predicted = np.array(predicted)
true = np.array(true)


confusion = confusion_matrix(true, predicted)

confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots()
im = ax.imshow(confusion)

fig.set_size_inches(9, 7.5)

labels = ['O', 'B-ORG', 'B-PER', 'I-PER', 'B-LOC', 'B-DATE', 'B-PRO', 'I-ORG', 'I-DATE', 'I-PRO', 'B-EVENT', 'I-EVENT', 'I-LOC']

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))

ax.set_xticklabels(labels,rotation='vertical')
ax.set_yticklabels(labels)

plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Confusion matrix for Digitoday test set')
plt.colorbar(im)
plt.savefig('confusion_digitoday.png', dpi=400)
fig.tight_layout()