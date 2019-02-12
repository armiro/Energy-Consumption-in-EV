# Driving Range Estimation and Energy Consumption Rate Deviation Classification in Electric Vehicles using Machine Learning Methods

Using a dataset collected from 
https://spritmonitor.de/ 
driving range for electric vehicles
is predicted via input features, such as driving_style, avg_speed and route_type.

* Regressors:
1) Linear Regression
2) Multilayer Perceptron (MLP)
3) Random Forest
4) AdaBoost
* Classifiers:
1) Support Vector Machines (SVM)
2) Multilayer Perceptron (MLP)
3) Random Forest

### Input data
Dataset crawler (```vehicle_crawler.py```) and 
an example result (```volkswagen_e_golf.csv```) in csv file can be found here:
https://github.com/armiro/crawlers/tree/master/SpritMonitor-Crawler 


### Run the code
First, change the dataset path in both the files. Then,
* run the ```driving_range_prediction.py``` file to predict the trip distance
of the electric vehicle; how long this vehicle can go in the next trip.
* run the ```ECR_deviation_classification.py``` file to classify the ECR
deviation from the manufacturer; whether in this trip ECR is higher
(more consumption) or lower (less consumption) than the factory-defined ECR.

