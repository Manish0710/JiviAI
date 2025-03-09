from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from .CONSTANTS import IMG_SIZE, BATCH_SIZE

input_shape = (224, 224, 3)  

def get_model():

    model = Sequential()


    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))


    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid')) 

    return model

# Define the learning rate scheduler function
def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:  # Halve LR every 10 epochs
        return lr * 0.5
    return lr


test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values
    validation_split=0.1    # Reserve 10% of the data for validation
)

# Training Generator (90% of data)
train_generator = train_datagen.flow_from_directory(
    'dataset_augmented/',   
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE,  
    class_mode='binary',    
    shuffle=True,           
    subset='training'       # Use 90% of the data for training
)

# Validation Generator (10% of data)
validation_generator = train_datagen.flow_from_directory(
    'dataset_augmented/',   
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE,  
    class_mode='binary',    
    shuffle=False,          # No need to shuffle validation data
    subset='validation'     # Use 10% of the data for validation
)


test_generator = test_datagen.flow_from_directory(
    'test/',
    target_size=IMG_SIZE, # Resize images
    batch_size=BATCH_SIZE,  # Define batch size
    class_mode='binary',     # Binary classification
    shuffle=True,           # Ensures reshuffling               
)