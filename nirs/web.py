import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_raw = pd.read_csv('breast_cancer.csv', sep=";")
data = data_raw.drop_duplicates()

data_X_train, data_X_test, data_Y_train, data_Y_test = train_test_split(data[['perimeter', 'concave points', 'compactness']].values, 
    data['diagnosis'].values, test_size=0.5, random_state=1)

scale_cols = ['perimeter', 'concave points', 'compactness']

# Функция для создания и обучения модели
def train_model(n_estimators, max_depth):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(data_X_train, data_Y_train)
    return model

st.title("Демонстрация модели случайного леса")
st.write("Настройте гиперпараметры модели и посмотрите на её производительность")

n_estimators = st.slider("Количество деревьев", 1, 100, 10)
max_depth = st.slider("Максимальная глубина дерева", 1, 20, 5)

# Обучение модели с заданными гиперпараметрами
model = train_model(n_estimators, max_depth)

# Оценка модели
y_pred = model.predict(data_X_test)
accuracy = accuracy_score(data_Y_test, y_pred)

# Отображение результатов
st.write(f"Точность модели: {accuracy:.2f}")

# Выводим важности признаков
st.write("Важности признаков:")
feature_importances = model.feature_importances_
for feature, importance in zip(scale_cols, feature_importances):
    st.write(f"{feature}: {importance:.2f}")