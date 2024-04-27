![Status: Final](https://img.shields.io/badge/Status-Final-yellowgreen.svg)
[![license](https://img.shields.io/badge/DOI-10.1109/ICSPIS48872.2019.9066042-3498db)](https://ieeexplore.ieee.org/abstract/document/9066042)

# Driving Range Estimation and Energy Consumption Rate Deviation Classification in Electric Vehicles using Machine Learning Methods

### Description
Using a dataset collected from 
https://spritmonitor.de/ 
driving range for electric vehicles
is predicted via input features, such as `driving_style`, `avg_speed` and `route_type`.

* Regressors:
1) Linear Regression
2) Multilayer Perceptron (MLP)
3) Random Forest
4) AdaBoost
5) Deep Multilayer Perceptron (Deep MLP)
* Classifiers:
1) Support Vector Machines (SVM)
2) Multilayer Perceptron (MLP)
3) Random Forest
4) Deep Multilayer Perceptron (Deep MLP)

### Citation
Find the related published conference paper [here](https://ieeexplore.ieee.org/abstract/document/9066042).
```
@inproceedings{amirkhani2019electric,
  title={Electric Vehicles Driving Range and Energy Consumption Investigation: A Comparative 
  Study of Machine Learning Techniques},
  author={Amirkhani, Abdollah and Haghanifar, Arman and Mosavi, Mohammad R},
  booktitle={2019 5th Iranian Conference on Signal Processing and Intelligent Systems (ICSPIS)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
```

### Input data
Dataset crawler (```vehicle_crawler.py```) and 
an example result (```volkswagen_e_golf.csv```) in csv file can be found here:
https://github.com/armiro/crawlers/tree/master/SpritMonitor-Crawler 


### Run the code
First, change the dataset path in both files. Then,
* run the ```driving_range_prediction.py``` file to predict the trip distance
of the electric vehicle; how long this vehicle can go in the next trip.
* run the ```ECR_deviation_classification.py``` file to classify the ECR
deviation from the manufacturer; whether in this trip ECR is higher
(more consumption) or lower (less consumption) than the factory-defined ECR.

