**London Weather**

_Creator: Mathew Thomas_

**Introduction**

An EDA was performed as part of a Capstone project at BrainStation's Diploma Data Science Program. Machine learning techniques were then explored as well to predict future mean temperatures in London. 
 
**About**

EDA first used Pearson correlation to better understand the relationship between the variables. The mean temperature was then used as the the dependent variable.  Iterations of a linear regression model were used to help provide the best predictions for 'mean_temp'. It was noted there was alot of variance when comparing the residuals especially regarding the outliers.

I used a linear regression model to address the question and explore the linear relationships to 'mean_temp'. However, the next step would be to also address the outliers to improve accuracy. Additionally, we should also try to understand the non-linear relationships with mean_temp' through the usage of other models. A combinations of multiple models addressing both the linear & non-linear relationships would provide us the best accuracy in understand the trend of mean_temp' in London. 


tyjrtehrgefd


**The User**

Insights from these models would provide information necessary for decision-making to a variety of stakeholders. Some of these include:

1. Meteorologist agencies - provide weather forecasters tangible insights to report to the general public or industries so that respectively they can plan accordingly.
2. Energy companies - predictions help in establishing a plan for the distribution of energy accordingly with regards to their heating and cooling systems.
3. Urban planners - helps focus their attention and allocate resources accordingly towards the development of infrastructure as well as mitigate for potential disasters. 


**The Big Idea**

Based on historical data, and assuming the general trend for the linear & non-linear relationships are continuous with respect to the dependant variable, the mean temperatures at London, can be predicted with the best accuracy. The final model could in turn be used as a baseline for building similar models for other cities as well. 

**The Impact**

Accurate predictions of mean temperature can significantly impact decision-making across various sectors, leading to better planning for not only industries but also the general public. It can help reduce and mitigate risks associated with weather-related events. It can also increase optimization and efficiecy providing tanglible insights on how one can improve current systems in place. This in turn would improve on sustainable resource management with things such as water and energy.

**Dataset**

The dataset being utilised to perform the EDA was obtained from kaggle and retreived by _Emmanuel F. Werr_. It is an aggregate of different weather attributes extracted from the _European Climate Assessment & Dataset_ (ECA&D). The measurements were reported at a weather station near London's Heathrow airport. 
The dataset comprises of historical data collected from Jan 1st 1979 to Dec 31st 2020. It has 15341 rows and 10 columns. The models will use the variables needed to understand their relevant impact. This should provide enough variables for the models to understand their relevant impact.

date - recorded date of measurement

cloud_cover - cloud cover measurement in oktas

sunshine - sunshine measurement in hours (hrs)

global_radiation - irradiance measurement in Watt per square meter (W/m2)

max_temp - maximum temperature recorded in degrees Celsius (°C)

mean_temp - mean temperature in degrees Celsius (°C)

min_temp - minimum temperature recorded in degrees Celsius (°C)

precipitation - precipitation measurement in millimeters (mm)

pressure - pressure measurement in Pascals (Pa)

snow_depth - snow depth measurement in centimeters (cm)

**References**

F.Werr, E. "London Weather Data" December 2021. 
https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data

ECA&D. 
https://www.ecad.eu/dailydata/index.php



