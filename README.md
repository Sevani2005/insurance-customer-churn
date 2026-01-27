Insurance Customer Churn Prediction

I've built this machine learning project to help insurance companies figure out which customers might cancel their policies soon. By digging into customer habits and policy details, the model spots at-risk folks early, so teams can reach out with personalized offers or support to keep them onboard and boost retention.

Tech Stack:

Python 

Pandas & NumPy for data  processing

Scikit-learn for quick baseline machine learning models

LightGBM for super-fast, accurate predictions

Day 1 – Project Setup & Planning

Today, I started the project by setting everything up. I chose the insurance domain to predict customer churn and found a dataset for it. I also created a GitHub repository with a clean structure and added clear documentation to make future work easier.

What I did:
 -Chose the insurance domain for churn prediction
 -Found and reviewed a relevant dataset
 -Defined the project goal and purpose
 -Set up a GitHub repository with organized folders
 -Added clear notes in the README
 -Committed and pushed the initial setup to GitHub

Day 2 – Exploratory Data Analysis (EDA)

- Loaded the insurance churn dataset
- Understood the columns and data types
- Checked how many customers churned vs not churned
- Observed basic patterns and relationships in the data
- Got the data ready for model building

Day 3 – Model Development

On Day 3, I worked on building the churn prediction model using machine learning.
I focused on training the model and preparing it for future predictions.

What I did:
-Selected LightGBM algorithm for churn prediction
-split the dataset into training and testing sets
-Trained the model using the training data
-Saved the trained logic inside the src folder

Day 4 – Model Insights

On day 4, I worked on understanding how the churn prediction model behaves.

What I did:
 -Checked model performance
 -Looked at feature importance
 -Analyzed predictions and results
 -Added insights in model_insights.ipynb
Day 5 – Model Deployment & Serialization
On Day 5, I finalized the model training process and prepared it for production by saving the trained binary.

What I did:
- Refined the training script for better performance
- Evaluated the final LightGBM model metrics
- Serialized the trained model into `models/churn_model.pkl` for UI integration
- Organized the repository for application development

Day 6 – Interactive AI Dashboard
Today, I developed a high-performance web interface to make the model accessible to business users.

What I did:
- Built a multi-tab Streamlit dashboard
- Implemented real-time single customer churn prediction
- Developed a batch analysis tool for bulk processing customer lists
- Integrated Plotly visualizations for risk distribution and portfolio analytics
- Added automated business "Next Step" recommendations based on risk scores

Day 7 – Optimization & Final Polish
On the final day, I focused on making the application production-ready and professional.

What I did:
- Refined the UI to meet corporate design standards (Solid colors & Clean typography)
- Removed decorative elements to ensure a professional, hand-built aesthetic
- Optimized model loading logic using resource caching
- Finalized requirements.txt and project documentation for deployment
- Verified all features (Single & Batch Analysis) for 100% accuracy
