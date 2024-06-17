### Code of Automatic Depression Detection Network Based on Facial Images and Facial Landmarks Feature Fusion

### Introduction
```
models       Storing the model
```
```
load_data.py	Get the path to the image and match the tag to it.
dataset.py	Inherit torch.utils.Dataset, responsible for transforming data into iterators that torch.utils.data.DataLoader can handle.
train.py	model training
validate.py	verification model
test.py		Test the performance of the model and record the prediction scores, saved in testInfo.csv, which records the path, label, and prediction score of each image.
main.py		Model training entry file
```


### How to run the code
```
pip install -r requirements.txt
python main.py--->python test.py
```
