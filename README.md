# Disaster Response Pipeline Project
**The main objective of a project is to analyze the text messages using Natural Processing Processing and build a ML/NLP pipeline from the disaster data provided by Figure Eight and deploy the application using Flask.**

Github Link: https://github.com/shubhammahalank/Disaster_Response_Project-Data_Engineering-Udacity

### File Organization:
    .
    ├── app     
    │   ├── run.py                          
    │   └── templates   
    │       ├── go.html                      
    │       └── master.html                    
    ├── data                   
    │   ├── disaster_categories.csv           
    │   ├── disaster_messages.csv            
    │   └── process_data.py                  
    ├── models                   
    │   ├── classifier.pkl          
    │   ├── train_classifier.py   
    ├── README.md                   
    
    
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/