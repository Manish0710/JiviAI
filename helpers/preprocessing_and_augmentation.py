import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import shutil

from .CONSTANTS import input_root, output_root,IMG_SIZE, BATCH_SIZE

# Define dataset path
CATARACT_PATH = os.path.join(input_root, "cataract")
NO_CATARACT_PATH = os.path.join(input_root, "normal")


# Check dataset structure
print(f"Cataract images: {len(os.listdir(CATARACT_PATH))}")
print(f"No Cataract images: {len(os.listdir(NO_CATARACT_PATH))}")


def display_sample_images():


    # Load the image
    cataract_sample = os.path.join(CATARACT_PATH, os.listdir(CATARACT_PATH)[0])
    image_cataract = tf.io.read_file(cataract_sample)
    image_cataract = tf.image.decode_image(image_cataract)

    normal_sample = os.path.join(NO_CATARACT_PATH, os.listdir(NO_CATARACT_PATH)[0])
    image_normal = tf.io.read_file(normal_sample)
    image_normal = tf.image.decode_image(image_normal)

    # Convert to NumPy array
    cataract_array = image_cataract.numpy()

    # Display the image
    plt.imshow(cataract_array)
    plt.axis("off")
    plt.title('cataract')
    plt.show()

    # Convert to NumPy array
    normal_array = image_normal.numpy()

    # Display the image
    plt.imshow(normal_array)
    plt.axis("off")
    plt.title('normal')
    plt.show()



    # Get the range of pixel values
    min_value = np.min(cataract_array)
    max_value = np.max(cataract_array)

    print(f"Pixel Value Range cataract_array: Min = {min_value}, Max = {max_value}")

    # Get the range of pixel values
    min_value = np.min(normal_array)
    max_value = np.max(normal_array)

    print(f"Pixel Value Range normal_array: Min = {min_value}, Max = {max_value}")

    print(f"Image Shape contaract: {cataract_array.shape}") 
    print(f"Image Shape normal: {normal_array.shape}") 


# Define augmentation transformations
datagen = ImageDataGenerator(
    rotation_range=30,          # Rotate within 30 degrees
    width_shift_range=0.2,      # Shift horizontally by 20%
    height_shift_range=0.2,     # Shift vertically by 20%
    shear_range=0.2,            # Shear transformation
    zoom_range=0.2,             # Zoom in and out by 20%
    horizontal_flip=True,       # Flip image horizontally
    brightness_range=[0.8, 1.2] # Adjust brightness randomly
)

# Set dataset paths
os.makedirs(output_root, exist_ok=True)


def save_augmented_data():
# Process each class folder (cataract & normal)
    for class_name in ["cataract", "normal"]:
        input_class_dir = os.path.join(input_root, class_name)
        output_class_dir = os.path.join(output_root, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        # Iterate through all images
        for img_name in os.listdir(input_class_dir):
            img_path = os.path.join(input_class_dir, img_name)

            # Load image and ensure RGB format
            img = load_img(img_path, target_size=(224, 224))  
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Convert to batch format

            # Copy the original image to the new dataset
            shutil.copy(img_path, os.path.join(output_class_dir, img_name))

            # Generate augmented images
            aug_iter = datagen.flow(img_array, batch_size=1, save_to_dir=output_class_dir,
                                    save_prefix=f"aug_{class_name}_{img_name.split('.')[0]}", save_format="jpg")

            # Save 5 augmented images per original
            for i in range(5):  
                next(aug_iter)

    print("✅ Data augmentation complete! Original images and augmented images saved in 'dataset_augmented/'.")


