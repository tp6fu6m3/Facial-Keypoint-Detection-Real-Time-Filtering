# Facial Keypoint Detection Real-Time Filtering

This project is able to run on NVIDIA Jetson-nano device.
Competition website on [**Kaggle Facial Keypoint Detection**](https://www.kaggle.com/c/facial-keypoints-detection)

Built an end-to-end facial keypoint recognition system. Facial keypoints include points around the eyes, nose, and mouth on any face and are used in many applications, from facial tracking to emotion recognition. The completed project takes in any image containing faces and identifies the location of each face and their facial keypoints.

- Convolutional Neural Networks
- Facial Keypoint Detection
- OpenCV

## Quick Start

### Linux user

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/tp6fu6m3/Facial-Keypoint-Detection-Real-Time-Filtering.git
cd Facial-Keypoint-Detection-Real-Time-Filtering
```

2. Download the training and test data on Kaggle.
	- go to https://www.kaggle.com/
	- go to Account → Creat New API Token
	- put kaggle.json under ~/.kaggle/
	- go to https://www.kaggle.com/c/facial-keypoints-detection
	- join competition
```
pip3 install kaggle
kaggle competitions download -c facial-keypoints-detection
unzip facial-keypoints-detection.zip -d data
for z in data/*.zip; do unzip "$z" -d data; done
```

3. Install a few required pip packages (including OpenCV).
```
pip3 install -r requirements.txt
```

4. Train the model and save it as '.h5' format.
```
cd src
python3 main.py
```

5. Demonstrate the real time filtering with the well-trained model.
```
python3 demo.py
```
	- press 'q' to quit the program
	- press 'w' to previous filter
	- press 'e' to next filter

### Windows user

1. Download the repository, and navigate to the downloaded folder.
	- go to https://github.com/tp6fu6m3/Facial-Keypoint-Detection-Real-Time-Filtering
	- go to Code → Download ZIP
	- unzip Facial-Keypoint-Detection-Real-Time-Filtering-main.zip
	- cd Facial-Keypoint-Detection-Real-Time-Filtering-main

2. Download the training and test data on Kaggle.
	- go to https://www.kaggle.com/c/facial-keypoints-detection/data
	- Download ALL
	- unzip facial-keypoints-detection.zip under Facial-Keypoint-Detection-Real-Time-Filtering-main/data/

3. Install a few required pip packages (including OpenCV).
```
pip install -r requirements.txt
```

4. Train the model and save it as '.h5' format.
```
cd src
python main.py
```

5. Demonstrate the real time filtering with the well-trained model.
```
python demo.py
```
	- press 'q' to quit the program
	- press 'w' to previous filter
	- press 'e' to next filter
