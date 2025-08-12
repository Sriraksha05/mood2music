# evaluate.py
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = tf.keras.models.load_model('models/emotion_best.h5')
test_loss, test_acc = model.evaluate(test_gen)
y_true = test_gen.classes
y_prob = model.predict(test_gen)
y_pred = np.argmax(y_prob, axis=1)

print("Test accuracy:", test_acc)
print(classification_report(y_true, y_pred, target_names=['angry','disgust','fear','happy','sad','surprise','neutral']))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['angry','disgust','fear','happy','sad','surprise','neutral'], yticklabels=['angry','disgust','fear','happy','sad','surprise','neutral'])
plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()
