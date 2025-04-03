import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify

# Step 1: Load the dataset
data = pd.read_csv('ipl2024.csv')

# Step 2: Clean the data
data = data[~data['winner'].isin(['None', 'Abandoned'])]
data = data.dropna()

# Step 3: Preprocess the data
categorical_cols = ['team1', 'team2', 'toss_winner', 'decision', 'winner', 'most_runs', 'most_wkts']
data = pd.get_dummies(data, columns=categorical_cols)

X = data.drop(columns=['player_of_the_match', 'id', 'date'])  # Features
y = data['player_of_the_match']  # Target variable

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        match_data = request.json
        input_data = pd.DataFrame([match_data])
        input_data = pd.get_dummies(input_data, columns=categorical_cols)

        # Align input data columns with training data using pd.concat()
        missing_cols = list(set(X.columns) - set(input_data.columns))  # Convert set to list
        missing_data = pd.DataFrame(0, index=input_data.index, columns=missing_cols)  # Add missing columns
        input_data = pd.concat([input_data, missing_data], axis=1)  # Concatenate columns

        # Reorder columns to match the training data
        input_data = input_data[X.columns]

        # Make prediction
        prediction = model.predict(input_data)
        return jsonify({'player_of_the_match': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)