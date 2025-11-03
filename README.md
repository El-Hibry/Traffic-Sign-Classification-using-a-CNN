# Traffic-Sign-Classification-using-a-CNN
This project designs and implements a deep learning model for real-time traffic sign classification. It uses a Convolutional Neural Network (CNN) built with Python, TensorFlow, and Keras to classify images from the German Traffic Sign Recognition Benchmark (GTSRB) dataset.
## Aim of the Project
To design and implement a deep learning model for real-time traffic sign classification to enhance road safety and support the development of autonomous vehicle systems.
## 🚀 Project Objectives
- Construct a Convolutional Neural Network (CNN) architecture suitable for multiclass image classification.
- Train the model on a labeled dataset of traffic sign images.
- Test and evaluate the model's performance using key metrics such as accuracy, precision, and recall.
- Build the entire system using the Python programming language with the TensorFlow and Keras libraries.
  
  ## 🛠️ Technologies Used
- Python 3
- TensorFlow & Keras (for building and training the CNN)
- Pandas (for data handling and evaluation)
- NumPy (for numerical operations)
- Matplotlib & PIL (for image loading and visualization)
- Scikit-learn (for splitting the dataset)
- Kaggle API (for data acquisition)
 ### 📂 Project Workflow
1. **Setup:** The notebook begins by setting up the Google Colab environment to connect to the Kaggle API, install the necessary client, and download the gtsrb-german-traffic-sign dataset.
2. **Data Preprocessing:**
- The downloaded dataset is unzipped.
- All images from the 43 ```Train``` class folders are loaded.
- Image dimensions are analyzed, and all images are resized to a standard 50x50 pixels with 3 color channels (RGB).
- Pixel values are normalized from the [0, 255] range to the [0, 1] range for faster and more stable training.
- The full dataset is split into an **80% training set** and a **20% validation set**.
3. **Model Architecture:** A Sequential CNN model is constructed with the following layers:
- ```Conv2D``` (64 filters, 3x3 kernel, 'relu' activation, padding='same')
- ```MaxPool2D``` (2x2 pool size)
- ```Dropout``` (50%)
- ```Conv2D``` (64 filters, 3x3 kernel, 'relu' activation)
- ```MaxPool2D``` (2x2 pool size)
- ```Dropout``` (50%)
- ```Flatten```
- ```Dense``` (128 neurons, 'relu' activation)
- ```Dropout``` (50%)
- ```Dense``` (43 neurons, 'softmax' activation) - The output layer, one for each class.
4. **Training:**
- The model is compiled with the ```adam``` optimizer and ```sparse_categorical_crossentropy``` loss function.
- The model is trained for 10 epochs with a batch size of 128, achieving a validation accuracy of ~98.6%.
5. **Evaluation & Testing:**
- The training and validation accuracy/loss are plotted to visualize performance.
- The separate ```Test``` dataset is loaded, preprocessed using the same scaling function.
- The trained model is used to predict the classes for the test images.
- Finally, several predictions are compared against the true labels from ```Test.csv``` to confirm the model's real-world performance.
## How to Run
1. **Get Kaggle API Key:**
  - Go to your Kaggle account, click on your profile picture, and select "Account".
  - Scroll down to the "API" section and click "Create New API Token".
  - This will download a ```kaggle.json``` file.
2. **Run in Google Colab:**
- Open the ```.ipynb``` file in Google Colab.
- Run the second cell (the one with ```files.upload()```).
- Upload the ```kaggle.json``` file you just downloaded.
3. **Execute Cells:** Run all subsequent cells in order to install dependencies, download the data, preprocess it, build the model, and train it.
  
## 📋 Step-by-Step Code Explanation
Here is a detailed breakdown of what each section of the code does.

1. **Setup and Data Download (Cells 2 - 8)**
This section prepares the environment by downloading the dataset from Kaggle.

- **Cell 2-4:** These cells handle the Kaggle API setup. ```files.upload()``` prompts you to upload your ```kaggle.json``` API key. The key is then installed, moved to the correct directory (```~/.kaggle/```), and its permissions are set.
- **Cell 5-7:** A new directory ```traffic_sign_dataset``` is created. The code then searches Kaggle for the "gtsrb-german-traffic-sign" dataset and downloads the specific one by ```meowmeowmeowmeowmeow```.
- **Cell 8:** This cell unzips the downloaded file. It's important to note that the zip file contains multiple data folders (e.g., ```train``` and ```Train```). Your code correctly removes the redundant lowercase ```train``` folder and unused ```Meta``` folders, keeping the primary ```Train``` (uppercase) and ```Test``` (uppercase) directories, which are used for the project.

2. **Data Loading and Preprocessing (Cells 9 - 19)**
This section loads, cleans, and prepares the images for the neural network.
- **Cell 9:** Imports all necessary libraries like ```os```, ```pandas```, ```numpy```, ```matplotlib```, ```tensorflow```, and ```sklearn```.
- **Cell 10:** Plots 16 random images from the ```Test``` folder to give you a first look at the data you'll be working with.
- **Cells 11-12:** This is a data exploration step. The code loops through all 39,209 images in the ```Train``` directory (across all 43 sub-folders) to find their dimensions. It then calculates the mean height and width, which come out to be approximately **50x50 pixels**. This justifies the decision to resize all images to this standard size.
- **Cell 13:** This is the main preprocessing loop. It iterates through all 43 class folders (0 to 42):
1. It opens each image using ```PIL.Image.open```.
2. Resizes every image to **(50, 50)**.
3. Converts the image to a NumPy array.
4. Appends the image array to the ```images``` list and its corresponding class index (the folder name, ```i```) to the ```label_id``` list.
- **Cell 14:** Converts the ```images``` list into a single large NumPy array. The pixel values are then **normalized** by dividing by 255.0, scaling them from [0-255] to [0.0-1.0]. This helps the model train faster.
- **Cells 15-17:** Converts the ```label_id``` list to a NumPy array and checks the final shapes. It also uses ```value_counts()``` to show the number of images in each class, revealing that the dataset is imbalanced.
- **Cell 18:** Splits the data into a training set (for teaching the model) and a validation set (for checking its performance on unseen data). It uses an 80/20 split.
- **Cell 19:** This cell creates one-hot encoded versions of the labels (e.g., ```5``` becomes ```[0,0,0,0,0,1,0,...]```).
 **Note:** These variables (```y_train_cat``` and ```y_val_cat```) are not actually used in the final model training.
  
3. **CNN Model Definition and Training (Cells 20 - 23)**
This is the core of the project where the deep learning model is built and trained.
- **Cell 20:** This defines the CNN architecture using a ```Sequential``` model.
 1. **Conv2D (Layer 1):** The first convolutional layer scans the 50x50x3 image with 64 different 3x3 filters to find basic features like edges and curves.
2. **MaxPool2D (Layer 2):** Downsamples the image, reducing its size to 25x25 while keeping the most important features.
3. **Dropout (Layer 3):** Randomly "turns off" 50% of the neurons during training to prevent the model from memorizing the training images (overfitting).
4. **Conv2D & MaxPool2D (Layers 4-5):** A second convolutional/pooling block to learn more complex features from the output of the first block.
5. **Flatten (Layer 7):** Converts the 2D feature maps into a 1D vector so it can be fed into a standard neural network layer.
6. **Dense (Layer 8):** A fully connected layer with 128 neurons that combines all the learned features.
7. **Dense (Layer 10):** The final output layer with 43 neurons (one for each traffic sign class). It uses a ```softmax``` activation function to output a probability for each class.

- **Cell 21:** Compiles the model.
1 ```loss='sparse_categorical_crossentropy'```: This loss function is chosen because your labels (```y_train```) are simple integers (0, 1, 2...), not one-hot encoded arrays. This is why Cell 19 was unnecessary.
2 ```optimizer='adam'```: An efficient and popular optimizer.
3 ```metrics=['accuracy']```: Tells the model to report its accuracy during training.
- **Cell 22:** This is where the model is trained (```model.fit```) on the training data for 10 epochs (passes through the data). It validates its performance against the validation set (```x_val```, ```y_val```) after each epoch. The output shows a final validation accuracy of **98.6%**.
- **Cell 23:** Plots the ```accuracy``` vs. ```val_accuracy``` and ```loss``` vs. ```val_loss``` over the 10 epochs. The graphs show the model learned very effectively, with validation accuracy consistently rising and loss decreasing.

4. **Testing and Final Prediction (Cells 24 - 37)**
This section evaluates the trained model on the completely separate test dataset.
- **Cell 24:** Sets the ```test_path``` variable and removes the CSV file from that directory, as it's not an image.
- **Cell 25-26:** A ```scaling``` function is defined to resize and normalize the test images, just as was done for the training images. This function is then applied to all images in the ```Test``` folder.
- **Cell 27:** Loads the ```Test.csv``` file, which contains the correct ```ClassId``` (the true labels) for the test images. These are stored in ```y_test```.
- **Cell 28:** The model predicts the class for every image in ```test_images```. ```np.argmax``` is used to select the class index with the highest probability from the softmax output.
- **Cell 30:** A list ```all_lables``` is created to map the numeric class IDs (like ```1```) to their human-readable names (like "Speed limit (30km/h)").
- **Cells 31-37:** These cells are a final sanity check. I picked a few images from the test set (e.g., images at index 1, 9, and 10), display the image, and then print both the **Original Label** (from ```y_test```) and the **Predicted Label** (from ```y_pred```). In all cases shown, they match, confirming the model works!
