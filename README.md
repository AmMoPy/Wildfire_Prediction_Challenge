# Overview and objectives:
-	Create a machine-learning model capable of predicting the burned area in different locations across Zimbabwe over 2014 to 2017.
-	Predictions are the proportion of the burned area per area square, with values between 0 and 1.
- There are 533 area squares each with a unique ID ranging from 0 to 532
- Training data is aggregated on burned areas across Zimbabwe for each month since 2001 up to the end of 2013, there are 156 record(13 years *  12 months) for 533 areas for a total of ~83k records (156 * 533)

 This is a competition notebook, check [Leaderboard](https://zindi.africa/competitions/predict-fire-extent/leaderboard) 

# Evaluation:
-	Root Mean Squared Error `RMSE`.

# Modeling and Feature Engineering:
-	Overall objective and expected model behavior:
    -	My goal is to build a model that captures the signals from repetitive temporal, interactions and proximity features.

-	Modeling Overview:
    -	Running progressive 4 fold validation, re-training the model with each year addition and feature engineering within fold. `RMSE` will be calculated per fold and final evaluation will be based on Last fold score as it is most representative given full training data. Its expected to have a decreasing score per fold.

-	Modeling steps - per fold:
    -	Feature Engineering: 
        -	This includes experimenting with different combination of features and models, Experimenting included:
              -	Temporal features: given the fact that fire incidents peaks and repeats during specific months
              -	Features that capture change over time: this has been tested across all zones and within zones during the full training period and year on year. Examples: rolling mean, std, var, cov and z-scores using different rolling windows specially those near peak months.
              -	Feature interaction, ranges and ratios: for example, elevation multiply wind speed, Vapor pressure deficit divided Vapor pressure, max temperature - min temperature and so on.
              -	Feature grouping: grouping features then calculating dispersion and interaction statistics within groups. Grouping is done based on my understanding of what each feature represents and also experimenting with different group combinations. This has also been inspired by this paper [Meteorological Drought: Palmer](https://www.droughtmanagement.info/literature/USWB_Meteorological_Drought_1965.pdf)
              -	Discretization of continuous features using decision trees followed by optional encoding, when a decision tree makes a prediction, it assigns an observation to one of N end leaves, therefore, any decision tree will generate a discrete output, this creates a monotonic relationship with target variable. The choice of reordering the discretized interval using Ordinal encoding instead of OHE is to control the size of feature subset, existence of meaningful ordinal relationship (i.e.: temperature - high, low) and the fact that I'll be using tree based models which can handle ordinal values as split points.
              -	Performing winsorization, capping feature values at specific values, these values can be parametric(z-scores) or non-parametric(IQR, MAD, quantiles)
              -	Target encoding followed by binning, the coise of bins was based on sqrt(N) where N is number of areas. This is a sort of clustering area based on historical burn records and it did actually improve model performance.
              -	Calculating distance(haversine) between each area using Lat and Long, the challenge was to select specific reference areas from which to calculate distance between all other areas. Initial choice was to use high burn areas as reference ones but segmenting all areas and calculating distance between highest and lowest burn area within the segment did provide better results same as converting distance from KM to Miles. Segmentation is done as discussed in EDA section.
              -	Calculating year on year change in land cover, using 2001 as reference year.
              -	Scaling all features: My initial thought was to perform Power Transformation or MinMax Scaling but normalizing selected features to zero mean and unit variance using StandardScaler did provide better  results
              -	Target transformation: the only transformation that did improve performance given this feature set was log(1 + `burn_area`) as it is full of zeros. The choice of transformation was to unmask linear relationships between the features and target.

    -	Rational behind model selection:
        -	Using gradient boosting framework that relies on models that are robust to skewed distribution and absence of non-linear trends
        -	limiting initial hyper parameter optimization in favor of later voting; Fine tuning the final best candidate models
        -	Once an initial model is selected, additional models will be added to form a voting ensemble meta-estimator that fits all base models, each on the whole dataset, then averaging the individual predictions to form a final prediction.
        -	Using `lgb` as base model as it provided best initial results in terms of `rmse` and training speed; emphasis is placed on memory usage and training speed to trial with different feature subsets and transformations.
        - Additional models added to the voting estimator are `ExtraTreesRegressor`, `KNeighborsRegressor`, `HistGradientBoostingRegressor`, `BaggingRegressor` with base `ExtraTreeRegressor` estimator. The choice of these models was based initially on improved prediction results without Fine Tuning and latter on the final results after Fine Tuning
        - Fine tuning each individual estimator was done manually and sequentially, this means tuning hyperparameter of first model in the pipeline, assessing prediction results in terms of reduction in `RMSE` and then moving to next model.
        - Choice of hyperparameters to tune is mainly confined to the structure(depth, leaf nodes), size(number of boosting trees, number of features used) and gradient optimization of loss function(learning rate)
        -	Fixed seed was used for all estimators to ensure consistency of results

    -	Run time (including training and prediction):
        -	`lgb` 4 fold validation without feature engineering: ~1 seconds per fold
        -	`lgb` 4 fold validation with feature engineering: ~4 seconds per fold
        -	`Voting Regressor` 4 fold validation with feature engineering: ~34 seconds per fold
        - `Voting Regressor` on full training data for final submission with feature engineering: ~50 seconds

    -	Setting thresholds:
        -	Predictions are clipped to be in the range of 0 and 1, thus negative predictions are eliminated. This is reflected in the `rmse` reported per each validation fold
        -	Predictions are set to nil for areas showing consistent zero burn during last three training years(i.e: area ID 1). This is reflected in final model predictions and submission. The rationale behind this step is the limited availability of external data and that current features does not adequately represent these areas; This has minor effect on reduction of overall `rmse`.

[My YouTube Channel](https://youtube.com/@ammopy)

![Model](https://github.com/AmMoPy/Wildfire_Prediction_Challenge/blob/main/Machine_learning_project_fire_area_predictions.jpg) 
