# train_emotion.py (high-level)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

model = build_model()
callbacks = [
    ModelCheckpoint('models/emotion_best.h5', monitor='val_accuracy', save_best_only=True, mode='max'),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
]

# compute class weights (if using ImageDataGenerator)
labels = train_gen.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weights
)
# after freezefit, unfreeze some base layers for fine-tuning:
base = model.layers[0]  # or find MobileNet block
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
fine_history = model.fit(train_gen, epochs=20, validation_data=val_gen, callbacks=callbacks, class_weight=class_weights)
