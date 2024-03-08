# How to run your custom TF Lite Model on Windows

[![Link to my YouTube video!](https://raw.githubusercontent.com/MarkChito/Malabay_Silao_Zacarias/main/thumbnail.png)](https://youtu.be/i8Z8LUIfGZg)

Link to the video: [Malabay_Silao_Zacarias Machine Learning Project (How to run your custom TF Lite Model on Windows)](https://youtu.be/i8Z8LUIfGZg)

## Prerequisites
- Install [Git for Windows](https://git-scm.com/downloads) on your Windows Machine
- Install [Anaconda](https://www.anaconda.com/download) on your Windows Machine

## Setup
- Open Command Prompt by typing `win + R`,  then type `cmd` and press `Enter` on your Windows Machine
  
- Change directory to your "Desktop Folder" by typing the code below:
```
cd Desktop
```

- Clone the repository by typing the code below:
```
git clone https://github.com/MarkChito/Malabay_Silao_Zacarias.git
```

- Change directory to your repository:
```
cd Malabay_Silao_Zacarias
```

- Create a virtual environment with python 3.9
```
conda create --name Malabay_Silao_Zacarias python=3.9
```

- Activate the virtual environment:
```
activate Malabay_Silao_Zacarias
```

- Install the requirements:
```
pip install tensorflow opencv-python protobuf==3.20.*
```

- You can now test the model by typing the code below:
```
python image_detection.py --modeldir=model --imagedir=images/test
```

- **Note:** you can test other images by changing the `--imagedir` folder.

## Run it as Web Server
- Run the Web Server using this code:
```
python web_server.py
```

- Check the command line results, check this lines:
```
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:3000
 * Running on http://192.168.x.x:3000 -> Focus on this third line, this is the one you will use later.
```

- Open the browser of your mobile phone (server and phone must be connected to the same network), then type the address on the third line shown on your command line.

- On the user interface, click `Try it!`, upload an image abd click the button `Upload and Detect`.

- **Note:** The results may not be 100% accurate on NON-CHURCH images, it may detect a wrong result, so make sure you will upload images with church.