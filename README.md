# ML Classification Of Histological Images

This dataset comes from here: Kather, Jakob Nikolas. (2019). Histological images for MSI vs. MSS classification in gastrointestinal cancer, FFPE samples. Zenodo. http://doi.org/10.5281/zenodo.2530835

Much of the information in the description come either from the dataset description or the scientific article using it to predict MSI:

Microsatellite instability determines whether patients with gastrointestinal cancer respond exceptionally well to immunotherapy. However, in clinical practice, not every patient is tested for MSI, because this requires additional genetic or immunohistochemical tests.

Content
This repository contains 192312 unique image patches derived from histological images of colorectal cancer and gastric cancer patients in the TCGA cohort (original whole slide SVS images are freely available at https://portal.gdc.cancer.gov/). All images in this repository are derived from formalin-fixed paraffin-embedded (FFPE) diagnostic slides ("DX" at the GDC data portal). This is explained well in this blog: http://www.andrewjanowczyk.com/download-tcga-digital-pathology-images-ffpe/

The dataset has been splitted into train, val, and test using the split_folders package.

Three different models have been applied to the dataset to compare the accuracy of the models.

1. Conventional CNN (Convolutional Neural Network) Model (code_CNN.py)
2. ResNet50 (code_ResNet50.py)
3. VGG16 (code_VGG16.py)

ResNet50 and VGG16 are both deep learning models used for image classification tasks.

1. Conventional CNN:

Test Loss: 0.757960855960846
Test Accuracy: 0.6947954297065735

2. ResNet50:

Test Loss: 0.6655075550079346
Test Accuracy: 0.609785258769989

3. VGG16:

Test Loss: 0.5571413636207581
Test Accuracy: 0.7199605107307434

## VGG16 performs the best among the three models, followed by conventional CNN and ResNet50.

The plots depicting the training and validation loss for all the three models used for classiication of the histological images have been provided in this repository.





