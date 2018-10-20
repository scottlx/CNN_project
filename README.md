# CNN_project
## EC601 project2  
A simple image recognition classfier using CNN, Keras and Tensorflow backend on Google Colab  
## Dataset  
[Flower Recognition (Kaggle)](https://www.kaggle.com/alxmamaev/flowers-recognition) (only use daisy and dandelion)  
Test data are downloaded randomly from Google image  
## Directory  
    /content/drive/My Drive/flowers/
        train/
            daisy/
                daisy1.jpg
                daisy2.jpg
                ...
            dandelion/
                dandelion1.jpg
                dandelion2.jpg
                ...
        validation/
            daisy/
                daisy1.jpg
                daisy2.jpg
                ...
            dandelion/
                dandelion1.jpg
                dandelion2.jpg
                ...  
    /content/drive/My Drive/flowers_test/
        train/
            daisy/
                daisy1.jpg
                daisy2.jpg
                ...
            dandelion/
                dandelion1.jpg
                dandelion2.jpg
                ...
        validation/
            daisy/
                daisy1.jpg
                daisy2.jpg
                ...
            dandelion/
                dandelion1.jpg
                dandelion2.jpg
                ...
    
## Run  
1. Download the dataset and save in Googledrive  
2. Create a Colab notebook in your Googledrive
3. Enable the TPU accelerator
4. Run

## Result  
val_acc: 0.8646

Filename	              Predictions  
daisy/daisy1.jpg	         0  
daisy/daisy2.jpg	         0  
daisy/daisy3.jpg	         0  
daisy/daisy4.jpg	         0  
daisy/daisy5.jpg	         0  
daisy/daisy6.jpg	         1  
daisy/daisy7.jpg	         0  
daisy/daisy8.jpg	         0  
dandelion/dandelion.jpg	   1  
dandelion/dandelion1.jpg	 1  
dandelion/dandelion2.jpg	 1  
dandelion/dandelion3.jpg	 1  
dandelion/dandelion4.png	 1  
dandelion/dandelion5.jpg	 1  
dandelion/dandelion6.jpg	 1  
dandelion/dandelion7.jpg	 1  
