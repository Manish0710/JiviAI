# JiviAI
Assignment on Cataract Predictions

To run the project:
1. run API/run_api.py, by replacing ../test/normal/image_280.png with the correct image path (line 4)

To get the augmented data:
1. run run_data_aug.py

To train the model:
1. run train_model.py


models folder has the model weights saved as .h5 file.
logs folder has the tensorboard graph for training and validation. To run it use: tensorboard --logdir=logs/ and follow the hyperlink