import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from openpyxl import load_workbook

# Load the Excel file
excel_file_path = r'C:\Users\DELL\Documents\project\data.xlsx'
wb = load_workbook(excel_file_path)
sheet = wb.active
data = sheet.values
columns = next(data)
df = pd.DataFrame(data, columns=columns)


# Load your DataFrame from the Excel file
df = pd.read_excel('data.xlsx', engine='openpyxl')


# Read Excel file
df = pd.read_excel(excel_file_path, engine='openpyxl')

# Read the data from a specific sheet 
sheet_name = 'Sheet1'
df = pd.read_excel(excel_file_path, sheet_name, engine='openpyxl')

# Organize data into hierarchical structure
data_hierarchy = []

print(df.iterrows)
for index, row in df.iterrows():
    # print(row)
    # print(f"index:{index}")
    print(row)
    # instance = {
    #     'stand 1': {
    #         'stem 1': {'leaf size (mm)': row['leaf size (mm)'], 'stem size (mm)': row['stem size (mm)'], 'stem size (mm)': 0},
    #         'stem 2': {'leaf size (mm)': row['leaf size (mm)'], 'stem size (mm)': row['stem size (mm)'], 'stem size (mm)': 0},
    #         'yield (kg)': row['yield (kg)']
    #     }
    # }
    # data_hierarchy.append(instance)

# Extract features and target

# features = []
# target = []

# for instance in data_hierarchy:
#     a_data = instance['stand 1']
    
#     # Features under stem 1 and stem 2
#     features_b = []
#     features_e = []
#     features_f = []
#     for b_key, b_value in a_data['stem 1'].items():
#         if isinstance(b_value, dict):
#             for e_key, e_value in b_value.items():
#                 features_e.append(e_value)
#         else:
#             features_b.append(b_value)
#     for c_key, c_value in a_data['stem 2'].items():
#         features_f.append(c_value)
#     features_b.extend(features_e)
#     features_b.extend(features_f)
#     features.append(features_b)
    
    
#     # Target under stand 1
#     target.append(a_data['yield (kg)'])

# X = np.array(features)
# y = np.array(target)

# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train and evaluate models
# models = {
#     'Decision Tree': DecisionTreeRegressor(random_state=42),
#     'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#     'XGBoost': xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
#     'Support Vector Machine': SVR(kernel='linear')
# }

# for model_name, model in models.items():
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, predictions))
#     print(f"{model_name} RMSE: {rmse:.2f}")
    
