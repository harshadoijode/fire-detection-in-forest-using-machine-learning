import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
input_shape = (150, 150, 3)
batch_size = 32
epochs = 5
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('dataset/train', target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory('dataset/test', target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical')
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_generator.num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint('keras_Model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[checkpoint])
label_names = sorted(os.listdir('dataset/train'))
with open('labels.txt', 'w') as f:
    f.write('\n'.join(label_names))
