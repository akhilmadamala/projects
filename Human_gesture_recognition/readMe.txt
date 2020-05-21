## Link to the server ##
http://13.52.130.222:8000/server/predict/ (stopped running)

************************Web Framework ******************************************************
Used django restframework in amazon web services to host the server

************************Machine Learning Models*********************************************
Used Decision Trees, Logistic Regression, Random Forest, kNN as the 4 models. 

************************Data preprocessing**************************************************
Preprocessed the data by making even frames for all the data and then performing FFT on the data
for better results. Then used pickle to dump the model and used by the web server 

************************Server testing******************************************************
Mainly used postman software to send post requests and check the output data and used the 
debug server provided by the professor.

************************Accuracy scores on the sample test data*****************************
All the accuracy scores, precision, recall, confusion matrix are mentioned in the report evaluating the our ML models.
k fold validation 5 folds used

************************Running the service on the local system*******************************
pip install -r requirements.txt
python manage.py runserver 0.0.0.0:8000
