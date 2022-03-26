# **SER_project_ML_Approach**

This repo consists of :

- **SER_classification_task.ipynb** Jupyter notebook which is a Speech Emotion Recognition (SER) study on the Crema-D dataset.

  The **data** can be obtained from the following Kaggle link : **https://www.kaggle.com/ejlok1/cremad**
  
  In order to view the plotly plots of the ipynb file you may view it with "Jupyter nbviewer" : **https://nbviewer.org/**
  
- **report.pdf** which is a detailed report of the sections described in the ipynb file

- **presentation.pdf** which is a short summary of the study's main conclusions

- **demo** folder contains code to test new speech samples. The new sample must belong in one of the following classes : "happy", "neutral", "sad", "angry". 
  
  The folder contains the following files :

      - demo.py : code to test new samples
      - model.pickle : contains the learned model and the audio features to choose from the test sample
      - scaler.pickle : contains the learned standard scaler
      - "test_sample" folder : folder to put the wav file to test
   
   In order to test a new sample you may insert the wav file in the "test_sample" folder and then run the "demo.py" file.

- **requirements.txt**

- **pickles** folder which contains all pickle files generated after running the ipynb file. They are included in the repo for saving purposes.
