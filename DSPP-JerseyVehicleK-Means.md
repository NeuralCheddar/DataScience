Data Science Project – Clustering using ‘All vehicles registered in Jersey’ dataset

1 – Executive Summary 

The aim of this project is to use machine learning predictive algorithms to try and identify any meaningful groups within the government of Jerseys open source dataset for vehicles in Jersey. The features used in the model were CO2, Age (of vehicle) and CC.  

K-means was chosen as the clustering algorithm as The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares. 

The model identified 4 meaningful clusters within the data. Further analysis determined that these clusters represented 
1.	Older/High Emission vehicles
2.	Mid age/moderate emission vehicles
3.	New/Standard Emission Vehicles
4.	Zero/ultra low emission vehicles

By identifying groups of similar vehicles, policymakers and businesses may gain a better understanding of vehicle ownership trends, emissions profiles and technology adoption. Potential stakeholders include Jersey Government transport planners, environmental policy teams and commercial vehicle organisations. Potential use cases include supporting electric vehicle adoption strategies, informing environmental policy decisions, targeting communications to specific vehicle segments and assisting commercial organizations involved in vehicle sales, servicing or fleet management.

Recommendation is to compare HDBSCAN and other clustering models and estimate missing CO2 values using age, engine size and fuel type before rerunning the analysis

2 - Methods 

2.1 - Data collection 

The data was sourced from the Jersey government open data website. Link to the dataset:  All Vehicles Registered in Jersey - Datasets - Government of Jersey Open Data. 

File: All registered vehicles to end June 2026.csv. The dataset has 7 columns and 131036 rows

The dataset was chosen due to its size and suitability for clustering analysis. 

2.2 - Data preprocessing

The project used Excel for some of the data transformation. The age column was created from the ‘REG'D IN JERSEY’ column using the formula =IF(D2="",0,DATEDIF(D2,TODAY(),"Y")). Blank cells were assigned zero to avoid processing errors. 

CO2 was missing for many of the older vehicle data. These records were dropped from the data. The dataset now has 55363 records. 

The data was then saved as Cleansed_vehicle_data.csv and imported into Amazon Sagemaker AWS. 

Data was checked for null values using python. 

 
Figure 1: using .info() to check for null values

 
Figure 2: Changing data type of CAR_AGE column to integer for the model 

Prior to model development, CO₂ and vehicle age were standardized using StandardScaler to avoid variables measured on larger scales disproportionately influencing centroid placement

 
Figure 3: Visually checking the data for clusters. 

No clear clusters are initially visible in Figure 3, although zero-emission vehicles appear distinct.

2.3 - Data analysis

DBSCAN and HDBSCAN were considered but K-Means was selected due to its scalability and interpretability. Hierarchical clustering was discounted due to computational cost

For the model, a K-means clustering algorithm has been selected because the dataset consisted primarily of continuous numerical variables and contained over 55,000 observations following preprocessing. It is considered one of the most powerful and popular data mining algorithms in the research community. …such a clustering algorithm requires the number of clusters to be defined beforehand (Ahmed, Seraj and Islam, 2020). One way to find the ideal amount of clusters is using the 
elbow method. The cluster number obtained by using the Elbow method is a subjective result because it is a visual method, and does not provide a measurement metric to show which elbow point is explicitly the optimum (Shi et al., 2021)

Running the elbow method on the data shows that potentially, the ideal number for K is around 4. 

 
Figure 4 – The Elbow plot. This is showing that the ideal number of clusters is 4 

Another method of identifying the best number of clusters is using the silhouette method. It combines the two factors of cohesion and resolution. Cohesion is the similarity between the object and the cluster. When compared to other clusters, it is called separation. This comparison is achieved by the value of the Silhouette, which is in the range −1–1. The Silhouette value is close to 1, indicating that there is a close relationship between the object and the cluster (Yuan and Yang, 2019).


Here it shows that between range 3 and 5 could be the best number of clusters. 3 was tried but there was not enough business context. When re-run, the elbow plot showed the most meaningful groupings in the data when K=4.
 
Figure 5: Silhouette method for best fit. Showing 3 or 4 as being the best number of clusters. 




 
Figure 6: Model run with 4 clusters as advised by the elbow plot and silhouette method.

In relation to figure 6, it seems the purple outliers are driving the data to be grouped close together and throwing off the scale so outliers above 600 CO2 were removed to get a better fit for the model and to make it more interpretable. 

 
Figure 7: The final output of the clustering model showing 4 clusters on the adjusted scale after removing the outliers. 

Outliers were removed which improved visual separation, although the mid-range clusters continued to overlap. Engine size was therefore added as a third feature. 

 
Figure 8: 3D chart adding CC as another variable. 

The analysis of the clusters identified 4 potentially meaningful groups in the data which are as follows: 

1.	Older/High Emission vehicles
2.	Mid age/moderate emission vehicles
3.	New/Standard Emission Vehicles
4.	Zero/ultra low emission vehicles

Electric Vehicles (EV’s) have 0CC so adding this has helped clearly separate themselves from the low-CO2 petrol vehicles. 

Large-engine vehicles (High CC) correlate highly with CO2 but aren’t really distinguishable by age. 

Whilst the model identified several meaningful vehicle groups, there are some limitations that should be considered when interpreting the results.

Firstly, a large proportion of records were removed due to missing CO₂ emissions data. These missing values were mostly associated with older vehicles and therefore may introduce sampling bias into the analysis. The resulting clusters are representative only of vehicles for which emissions information was available.

Secondly, the clustering model was primarily built using vehicle age and CO₂ emissions. Whilst these variables were considered relevant for the project objectives, other potentially important features such as fuel type, vehicle class and make were not included within the final clustering model. Incorporating some of these additional variables may have produced additional clustering patterns.

K-Means assumes similarly shaped clusters and may not fully capture complex real-world vehicle populations. Alternative algorithms may identify different patterns.

Also, there is an issue where it is showing older cars as 0 CO2 and 0CC. Upon checking the vehicles in the data, it appears that this is a data quality issue.

Finally, the removal of extreme CO₂ outliers improved model interpretability but may have excluded a small number of legitimate vehicles from the final analysis.

These limitations should be considered when applying the findings to policy or commercial decision-making.


3 – Recommendations

The clustering analysis successfully identified four vehicle clusters based on age and emissions characteristics. These findings provide a foundation for further analysis and the potential for operational use.

The first recommendation is to investigate alternative clustering techniques such as HDBSCAN, DBSCAN and other models. Comparing outputs across multiple algorithms would help determine whether the identified clusters are robust and whether additional vehicle segments exist within the data.

Secondly, future work should focus on improving data quality by addressing missing CO₂ values. One potential approach would be to develop a regression model using variables such as fuel type, engine size and vehicle age to estimate missing emission records. This would increase the number of observations available to the clustering algorithm and may improve segmentation accuracy.

From a business perspective, the results could support government initiatives relating to environmental policy, electric vehicle adoption and charging infrastructure planning. The identified low-emission and zero-emission groups may help measure the effectiveness of sustainability programmes over time.

Finally, periodic rerunning of the clustering model would allow changes in vehicle ownership patterns and emissions profiles to be monitored, providing ongoing insights that can help drive future decision making

4 - Ethical, Privacy and Regulatory Considerations

The dataset used in this project was obtained from the Government of Jersey Open Data Portal and did not contain personally identifiable information as it was likely removed before publication online. As a result, privacy risks associated with individual vehicle owners were minimal.

Data quality also represents a potential source of bias. A significant proportion of older vehicles contained missing CO₂ emission values and were removed during preprocessing. This may result in the underrepresentation of certain vehicle categories within the final analysis.

To mitigate these risks, findings were validated using multiple visualization techniques, including two-dimensional and three-dimensional plots, and were assessed against real-world vehicle characteristics to ensure accurate interpretation. Any future operational use of the model should include ongoing monitoring of data quality and periodic validation of model outputs.



 
5 - References


Ahmed, M., Seraj, R. and Islam, S.M.S. (2020). The k-means Algorithm: A Comprehensive Survey and Performance Evaluation. Electronics, [online] 9(8), p.1295. doi:10.3390/electronics9081295. 

Shi, C., Wei, B., Wei, S., Wang, W., Liu, H. and Liu, J. (2021). A quantitative discriminant method of elbow point for the optimal number of clusters in clustering algorithm. EURASIP Journal on Wireless Communications and Networking, 2021(1). doi:10.1186/s13638-021-01910-w.

Yuan, C. and Yang, H. (2019). Research on K-Value Selection Method of K-Means Clustering Algorithm. J, 2(2), pp.226–235. doi:10.3390/j2020016. 


