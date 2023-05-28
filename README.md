# Disaster Response Pipeline Project

## Libraries used in this project:
<ol>
  <li>pandas</li>
  <li>sys</li>
  <li>sqlalchemy</li>
  <li>numpy</li>
  <li>typing</li>
  <li>sklearn</li>
  <li>nltk</li>
  <li>gensim</li>
  <li>re</li>
  <li>pickle</li>
  <li>json</li>
  <li>flask</li>
  <li>joblib</li>
</ol>


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Files in this repository:
<ol>
    <li>app/
        <ol>
            <li><b>run.py:</b> This is file contains the functionality for generating visualizations from the training set and tagging new messages 
            using the classifier. </li>
            <li>templates/
                <ol>
                    <li><b>master.html:</b> the template for visualizations and searching new messages.</li>
                    <li><b>go.html:</b> the template for tagging the related categories for new messages.</li>
                </ol>
            </li>
        </ol>
    </li>
    <li>data/
        <ol>
         <li><b>disaster_categories.csv:</b> This file contains the categories and unique identifier for each message.</li>
         <li><b>diaster_messages.csv:</b> This file contains the message data and the unique identifer for each message.</li>
         <li><b>DisasterResponse.db:</b> This database file contains the message and category data combined into a sqlite database</li>
         <li><b>process_data.py:</b> This file contains the functions for preprocessing the message and category data, and loading
         the combined data into a database.</li>
        </ol>
    </li>
    <li>models/
        <ol>
            <li><b>resample.py:</b> This file contains functionality for resampling the training set for improving class imbalance.</li>
            <li><b>tokenizer.py:</b> This file contains functionality for preprocessing and tokenizing text used for training the classification
            model.</li>
            <li><b>train_classifier.py:</b> This file contains the functionality for preprocessing, and training and serializing the classication model.</li>
            <li><b>classifier.pkl:</b> This file is the serialized classification model.</li>
        </ol>
    </li>
    <li>.DS_Store</li>
</ol>


## Notes on the Dataset (Class Imbalance)
One of the challenges with this dataset pertains to the class imbalance in many of the outputs in the dataset. For example, there are 25934 cases in the dependent variable 'fire' where 'fire == 0', yet only 282 instances where 'fire == 1' - this level of class imbalance has tendency to eliminate the effect of the minority class. Additionally, when optimizing for average weighted accuracy with severe class imbalance, the model can predict the majority class all or most of the time, and obtain a high level of accuracy. In the case of a disaster response pipeline, you could run into a scenario where you never predict when a disaster has occurred, but be highly accurate. 

One of the ways to resolve class imbalance, is to resample the training dataset, e.g., by upsampling the minority class or downsampling the majority class, however, resampling is complicated if not infeasible in a multi-class, multi-output problem. Another option is to change the approach of optimizing for weighted average accuracy or a weighted metric altogether, and instead considering an unweighted metric, such as unweighted average recall which will place more importance on the underrepresented class versus the weighted metric.


## Link to Repository: https://github.com/jmcollie/disaster_response_pipeline