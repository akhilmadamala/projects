## Link to the server ##<br/>
http://13.52.130.222:8000/server/predict/ (stopped running)

**Web Framework**<br/>
Used django restframework in amazon web services to host the server

**Machine Learning Models Used**<br/>
Used Decision Trees, Logistic Regression, Random Forest, KNN as the 4 models. 

**Data preprocessing**<br/>
Recorded videos for every sign in american sign language and converted them in to key points data which indicate the various reference points for different parts of human body<br/>
Preprocessed the data by making even frames for all the data by subtracting the fixed pints like nose etc and normalized with shoulder and hip length and then performed FFT on the datafor better results.<br/> 
Then used pickle to dump the model and used by the web server <br/>

**Server testing**<br/>
Mainly used postman software to send post requests and check the output data and used the 
debug server provided by the professor.

**Accuracy scores on the sample test data**<br/>
All the accuracy scores, precision, recall, confusion matrix are mentioned in the report evaluating the our ML models.
k fold validation 5 folds used

**Running the service on the local system**
pip install -r requirements.txt
python manage.py runserver 0.0.0.0:8000
