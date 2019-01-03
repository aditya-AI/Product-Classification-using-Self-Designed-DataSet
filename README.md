# Product
##### This project is all about creating your own dataset from scratch and making a robust classifier. There are in total four classes namely Coke Can, ThumbsUp Can, Mirinda Bottle, Tide Packet! The predictions of the network are in the `Results` folder.

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
