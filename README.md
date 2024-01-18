# London Weather

_Creator: Mathew Thomas_

## About

A number of weather attributes were cleaned and compiled together to form a London Weather dataset as part of a Capstone project at BrainStation's Diploma Data Science Program. Based on the historical data, a question was asked on how well we could predict quantifiable precipitation in London. With the aid of machine learning, we can try to predict this with the best accuracy. The final model could in turn be used as a baseline for building similar models for other cities as well. 

## Introduction

Precipitation is any liquid or frozen water that begins in the atmosphere and falls to the Earth's surface. Predicting it can provide cruical information to a wide variety of groups and entities. 
 A linear regression model was initially used to address the question and explore the linear relationships to 'mean_temp'.   
 

## The User

Insights from these models would provide information necessary for decision-making to a variety of stakeholders. Some of these include:

1. Meteorologist agencies - provide weather forecasters tangible insights to report to the general public or industries so that respectively they can plan accordingly.
2. Energy companies - predictions help in establishing a plan for the distribution of energy accordingly with regards to their heating and cooling systems.
3. Urban planners - helps focus their attention and allocate resources accordingly towards the development of infrastructure as well as mitigate for potential disasters. 
 

## The Impact

Accurate predictions of mean temperature can significantly impact decision-making across various sectors, leading to better planning for not only industries but also the general public. It can help reduce and mitigate risks associated with weather-related events. It can also increase optimization and efficiecy providing tanglible insights on how one can improve current systems in place. This in turn would improve on sustainable resource management with things such as water and energy.

## Dataset

The dataset being utilised to perform the EDA was obtained from kaggle and retreived by _Emmanuel F. Werr_. It is an aggregate of different weather attributes extracted from the _European Climate Assessment & Dataset_ (ECA&D). The measurements were reported at a weather station near London's Heathrow airport. 
The dataset comprises of historical data collected from Jan 1st 1979 to Dec 31st 2022. It has 15341 rows and 10 columns. The models will use the variables needed to understand their relevant impact. This should provide enough variables for the models to understand their relevant impact.

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

EDA first used Pearson correlation to better understand the relationship between the variables. The mean temperature was then used as the the dependent variable. Using Stepwise Regression, iterations of a linear regression model were used to help provide the best predictions for 'mean_temp'. It was noted there was alot of variance  with 'cloud_cover', 'snow_depth', 'year' & 'month' when comparing the different model residuals. 
The next step would be to also understand the non-linear relationships, perhaps try using median instead of mean to fill in null values, and address the outliers. Additionally, we should also try to understand all relationships with 'mean_temp' through the usage of other models. A combinations of multiple models addressing both the linear & non-linear relationships would provide us the best accuracy in understanding the trend of 'mean_temp' in London.

## References

ECA&D. 
https://www.ecad.eu/dailydata/index.php



