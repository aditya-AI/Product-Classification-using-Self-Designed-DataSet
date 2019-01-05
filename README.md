# Product Classification using Self Designed DataSet
##### This project is all about creating your own dataset from scratch and making a robust classifier. There are in total four classes namely Coke Can, ThumbsUp Can, Mirinda Bottle, Tide Packet! The predictions of the network are stored in the `Results` folder.

##### The model achieves ~97%-98% accuracy on the data stored in the `Results` folder with almost 100% accuracy on the training set.

##### The data was collected in two ways:
<ol>
  <li> <b>Clicking pictures from iPhone 6S plus and,</b></li>
  <li> <b>Recording videos from Laptop.</b> </li>
</ol>

### One of the key aspect while collecting the training data was to make sure all the products have similar backgrounds so that the network learns the meaningful and discriminative features of the products in the images and not the `Background`.

### Second most important part was to make sure that the distribution of the validation data is almost the same as the testing data. This helped me to boost the performance of my network a lot since I made the architectural and hyperparameter changes based on the validation data that had the distribution similar to what I would expect in a real-life scenario (testing environment).

### From the `Results` section it should be clear to you that the background in the images is quite diverse and the model is able to classify almost all of the products correctly.

<b>Link to the Video:</b> <a href="https://www.youtube.com/watch?v=r87KPuP5yBI&list=PLmMsypEcVPfYmxj3RvrmZ2lAz1OCQd-V6" target="_blank"> Product Classification using Self Designed DataSet Playlist</a>

<a href="https://www.youtube.com/watch?v=r87KPuP5yBI&list=PLmMsypEcVPfYmxj3RvrmZ2lAz1OCQd-V6/" target="_blank">Product Classification using Self Designed DataSet Playlist</a>

1. Install virtual environment
```bash
pip3 install virtualenv
```

2. Initiate virtual environment
```bash
virtualenv venv
```

3. Activate virtualenv
```bash
source venv/bin/activate
```

4. Install the requirements
```bash
pip3 install -r requirements.txt
```

5. To create your own dataset from videos, run the below command by specifying the video (from path) and frames (to path). You shall expect for a 1 min. video 60 frames.
```bash
python3 video_to_images.py
```
6. Both training and testing (on images) is in the `Classification.ipynb` notebook, which can be opened in a jupyter notebook. Otherwise, run the below command in your virtual environment. Make sure to specify correct training and validation data paths.
```bash
python3 Classification.py
```

7. To test the model in real-time using your laptop's web camera the `webcamera.py` script can be useful. Multi-threading is used to avoid the delay in reading each frame and model's prediction on each frame. Both `.ipynb` and `.py` files are present in the repository.
```bash
python3 Web_Camera.py
```
