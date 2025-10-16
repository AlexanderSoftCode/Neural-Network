from zipfile import ZipFile
import os 
import urllib
import urllib.request 

URL  =  'https://nnfs.io/datasets/fashion_mnist_images.zip' 
FILE  =  'fashion_mnist_images.zip' 
FOLDER  =  'fashion_mnist_images_L' 

if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE} ... ' )
    urllib.request.urlretrieve(URL, FILE)

print("Unzipping images...")
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)

print('Done!')

#All that this block of code does is install the file from the url.
#Then we are to download it to actually the same pwd as the one the file is ran on