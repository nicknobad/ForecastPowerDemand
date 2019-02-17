# Forecast Power Demand

Work in progress in terms of splitting the big script into modules and making it more functional

Prerequisites:
1. API keys for elexon and weatherbit
2. python packages


 At the moment the script works with Initial Demand Outturn. To train on the Final Demand Outturn data you have to download it
 
 Steps: 
 
 0. Inputs
 1. FUN for holidays, doy, moy, dow
 2. Configs
 3. Download and prepare (clean, process and format) the datasets
    2A. Demand
    2B. Weather
 4. Train, CV and Test
    4A. Simple CV
    4B. Predictions on train, val and test
    4C. ToDo: Plot best model preds in the test dataset vs real
    4D. ToDo: Tune several models and benchmark their performance
 5. Download weather forecast
 6. Download demand forecast from NG; Combine 6A with 6B
    6A. DA HHly demand Forecast from NG
    6B. OC2-14 Peak demand. Explode it to HHly.
 7. Merge Fct Demand and Weather (from Steps 5 and 6) for Step 8. Get all time vars and Holidays
    Forecast HHly demand for the next 2 weeks and plot it
