# Rideshare-Regression

The purpose of this project was to develop a simple dashboard for running and editing hyperparameters of various regression models on a dataset.

The dataset chosen pertains to rideshare services offered by Uber and Lyft in the Boston, MA area. The dataset can be loaded via the following link : https://drive.google.com/file/d/1_c8CFS6ADMKY3WM5nFhowG2nD2YU_5ZK/view?usp=share_link . 

To create the dashboard the streamlit python library was utilized. Within the dashboard, there are various pages with interactable features allowing for the creation of histograms, normal probability plots, correlation matrix, and various Regressors (linear, decision tree, random forest, support vector). The tuning of the plots and models simply require user text input for hyperparameter variables or toggling of explanatory variables to be incorporated in the model via drop down selection.


## How to Run

To recreate the project and get the application running on your own system all dependencies must be installed within your python environment.

Then, the the repository needs to be cloned and the csv file containing the dataset from the link above must be placed into the same directory (folder where repository was cloned).

Now all that needs to be done is to run one command within the command prompt : > streamlit run home.py
