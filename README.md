# Precipitation in London

_Creator: Mathew Thomas_

## About

A number of weather attributes were cleaned and compiled together to form a London Weather dataset as part of a Capstone project at BrainStation's Diploma Data Science Program. Based on the historical data, a question was asked on how well we could predict precipitation in London with the aid of machine learning models.  

## Introduction

Precipitation is any liquid or frozen water that begins in the atmosphere and falls to the Earth's surface. Predicting it can provide cruical information to a wide variety of groups and entities. 
    
### The User

Insights from these models would provide information necessary for decision-making to a variety of stakeholders. Some of these include:

1. Meteorologist agencies - provide weather forecasters tangible insights to report to the general public or industries so that respectively they can plan accordingly.
2. Energy companies - predictions help in establishing a plan for the distribution of energy eg: Hydroelectricity.
3. Urban planners - helps focus their attention and allocate resources accordingly towards the development of infrastructure as well as mitigate for potential disasters. 
4. Agriculture - help farmers with the loss of crops due to drought or flooding 

### The Impact

Accurate predictions of precipitation can significantly impact decision-making across various sectors, leading to better planning for not only industries but also the general public. It can help reduce and mitigate risks associated with weather-related events. It can also increase optimization and efficiecy providing tanglible insights on how one can improve current systems in place. This in turn would improve on sustainable resource management with things such as water and energy.

### Dataset

The dataset being utilised is an aggregate of different weather attributes extracted from the _European Climate Assessment & Dataset_ (ECA&D). The measurements were reported at a weather station near London's Heathrow airport. 
The dataset comprises of historical data collected from Jan 1st 1979 to Dec 31st 2022. It has 16071 rows and 11 columns. The models will use the variables needed to understand their relevant impact. 

### Data dictionary:
- `DATE`: recorded date of measurement
- `CC`: Cloud Cover, measurement in oktas 
- `HU`: Humidity, measurement in %
- `QQ`: Global Radiation, irradiance measurement in Watt per square meter (W/m2)
- `TX`: Temperature Maximum, maximum temperature recorded in degrees Celsius (°C)
- `TG`: Temperature Mean, mean temperature in degrees Celsius (°C)
- `TN`: Temperature Minimum, minimum temperature recorded in degrees Celsius (°C)
- `RR`: Precipitation, precipitation measurement in millimeters (mm)
- `PP`: Pressure, pressure measurement in Pascals (hPa)
- `SD`: Snow Depth, depth measurement in centimeters (cm)
- `SS`: Sunshine, measurement in hours (hrs)

## Summary

A variation of models were run of which the Tuned LGBM seemed to showcase the most promise in terms of usability. However, more data and information is required to enhance the model's ability to predict precipitation. Additionally, discussion with SMEs could further improve data validity. Another possible avenue to investigate would be in deep learning models such as DeepAR. Further exploration is required into both forecasting and optimization methods to capture such a complex feature. 


## Reference

ECA&D. 
https://www.ecad.eu/dailydata/index.php



