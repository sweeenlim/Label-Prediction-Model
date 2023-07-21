# Label Prediction Model

Machine Learning model to predict the label for contracts dataset.

Built using Flask.

## Environment and Tools
1. scikit-learn
2. pandas
3. numpy
4. flask
5. matplotlib
6. imblearn

## Installation
```
pip install -r requirements.txt
```
```
python model.py
```
```
python app.py
```
```
docker run -p 8000:5000 cherrypaca/sap_ml_model
```
## Structure
```model.py``` contains the code for the machine learning model to predict the label for the corresponding provisions.

```app.py``` contains FLASK APIs that receives provision details through GUI calls, gets the predicted label based on my model and returns it.

HTML and CSS contains the HTML template and CSS styling to allow user to enter the provision detail and displays the predicted label.