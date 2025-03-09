import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
from helpers.model import get_model, lr_scheduler, train_generator, validation_generator, test_generator

print("Training samples per class:", train_generator.samples)
print("Validation samples per class:", validation_generator.samples)

model = get_model()
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.AUC(name="roc_auc", curve="ROC")]  # Adding ROC-AUC
)

print(model.summary())

# Define Callbacks
log_dir = "logs/fit_" + time.strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_callback = ModelCheckpoint(
    "models/cataract_best.weights.h5",  # Save only the best weights
    monitor='val_accuracy',  # Monitor validation accuracy
    save_best_only=True,  # Save only the best model
    save_weights_only=True,  # Save only weights
    mode='max',  # Maximize validation accuracy
    verbose=1
)

# Define the callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

# Train Model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    callbacks=[tensorboard_callback, checkpoint_callback, lr_callback]
)


res = model.evaluate(test_generator, return_dict=True)
print(res)
