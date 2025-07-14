import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class TitanicModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.model = RandomForestClassifier()

    def load_and_prepare_data(self):
        if not os.path.exists(self.file_path):
            print(f"File not found: {self.file_path}")
            return

        self.data = pd.read_csv(self.file_path)

        # Select important columns
        columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        self.data = self.data[columns].dropna()

        # Encode 'Sex'
        self.data['Sex'] = self.data['Sex'].apply(lambda x: 1 if x == 'female' else 0)

        self.X = self.data.drop('Survived', axis=1)
        self.y = self.data['Survived']

    def train_model(self):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            self.model.fit(X_train, y_train)
            accuracy = self.model.score(X_test, y_test)
            print(f"\nModel trained successfully. Accuracy: {accuracy:.2f}\n")
        except Exception as e:
            print("Error training the model:", e)

    def save_model(self, filename="model.pkl"):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved as '{filename}'")
        except Exception as e:
            print("Error saving the model:", e)

    def show_visuals(self):
        print("\nGenerating data visualizations...")

        # Survival Count
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.data, x='Survived')
        plt.title('Survival Count (0 = No, 1 = Yes)')
        plt.tight_layout()
        plt.show()

        # Survival by Gender
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.data, x='Sex', hue='Survived')
        plt.title('Survival by Sex (0 = Male, 1 = Female)')
        plt.tight_layout()
        plt.show()

        # Survival by Class
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.data, x='Pclass', hue='Survived')
        plt.title('Survival by Passenger Class')
        plt.tight_layout()
        plt.show()

        # Age Distribution
        plt.figure(figsize=(6, 4))
        sns.histplot(data=self.data, x='Age', bins=30, kde=True)
        plt.title('Age Distribution of Passengers')
        plt.tight_layout()
        plt.show()

        # Age vs Fare
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=self.data, x='Age', y='Fare', hue='Survived')
        plt.title('Age vs Fare with Survival')
        plt.tight_layout()
        plt.show()

        # Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.data.corr(), annot=True, cmap='Blues')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()

def main():
    print("Titanic Survival Prediction Project Started\n")
    project = TitanicModel("titanic.csv")
    project.load_and_prepare_data()
    project.train_model()
    project.save_model()

    # Ask user if they want to see visuals
    see_charts = input("Do you want to see data visualizations? (y/n): ").lower()
    if see_charts == 'y':
        project.show_visuals()
    else:
        print("Visualization skipped.")

if __name__ == "__main__":
    main()
