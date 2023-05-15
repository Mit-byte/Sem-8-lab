import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Preprocess dataset
# Load, resize, normalize images, and split into training and testing sets

def create_model(input_shape):
    inputs = Input(input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    age_output = Dense(1, activation='linear', name='age_output')(x)
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)

    model = Model(inputs=inputs, outputs=[age_output, gender_output])
    model.compile(optimizer=Adam(lr=0.001), loss=['mse', 'binary_crossentropy'], metrics=['mae', 'accuracy'])
    return model

input_shape = (48, 48, 1)
model = create_model(input_shape)
model.summary()

# Train the model
callbacks = [
    ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss'),
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
]
history = model.fit(X_train, {'age_output': y_train_age, 'gender_output': y_train_gender}, batch_size=64, validation_split=0.2, epochs=50, callbacks=callbacks)

# Evaluate the model
model.evaluate(X_test, {'age_output': y_test_age, 'gender_output': y_test_gender})

def predict_age_gender(image, model):
    # Preprocess input image
    image = cv2.resize(image, (48, 48))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)

    age_pred, gender_pred = model.predict(image)
    age = int(np.round(age_pred[0]))
    gender = 'Female' if np.round(gender_pred[0]) == 0 else 'Male'
    return age, gender

# Test the prediction function
age, gender = predict_age_gender(test_image, model)
print("Predicted Age:", age)
print("Predicted Gender:", gender)
