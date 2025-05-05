# CPSC381 Final Project Submission

## Structure of Repository

### Inference + Training Files

webcam_hosted.py --> specify path to weights in the file and have a valid webcam, this program then hosts display of the model in action in your browswer using the wave_sequence_model_final.keras weights
drone_training.ipynb --> full final training pipeline for creating the model
drone_inference.ipynb --> test run where you feed in data and it outputs a prediction, using final weights file

### Downloading weights

1. Get the ID from the share link created when you attempt to share uploaded links from google drive: https://drive.google.com/file/d/ID/view

2. using the ID (note: it is between /view and /file/d) plug it into the command in 4. Repace "ID" with your ID from the gdrive link as specified in step 1.

3. run pip install gdown

4. gdown --id ID -O wave_sequence_model_final.keras

### Crucial: Data Preparation

Note that our dataset was created custom by fusing together larger datasets piecemeal and performing custom alterations to specifically suite our task of performing inference from a drone. Therefore, you need to download our dataset and correctly set up the path to this dataset in your own personal google drive so that the model can actually access it.
