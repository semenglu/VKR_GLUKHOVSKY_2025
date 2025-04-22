import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Загрузка данных
df = pd.read_csv('kur5.KEE.csv', sep=';')
df.columns = ['Возраст', 'Пол', 'Степень стеноза внутренней сонной артерии', 'Стенокардия ФК', 'ПИМ', 'Нарушения ритма', 'ХСН', 'ФК по NYHA', 'СД', 'ХОБЛ', 'ОНМК в анамнезе', 'Вмешательство', 'вид КЭЭ', 'ВПШ', 'летальность', 'Осложнения']


# Подготовка признаков и целевой переменной
X = df.iloc[:, :-1]  # все столбцы, кроме последнего
y = df.iloc[:, -1]   # последний столбец

# Преобразование категориальных данных
categorical_features = ['Пол', 'Степень стеноза внутренней сонной артерии',
       'Стенокардия ФК', 'ПИМ', 'Нарушения ритма', 'ХСН', 'ФК по NYHA', 'СД',
       'ХОБЛ', 'ОНМК в анамнезе', 'Вмешательство', 'вид КЭЭ', 'ВПШ',
       'летальность']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Стандартизация числовых данных
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Пайплайн преобразований
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Подготовка данных
X = preprocessor.fit_transform(X)
y = y.values  # Подготовка целевой переменной

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Определение модели
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Используем sigmoid для бинарной классификации
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

# Оценка модели на тестовых данных
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Точность модели: {accuracy:.2f}")


# Функция для GUI, которая использует модель
def submit():
    inputs = {}
    for col, entry in entries.items():
        value = entry.get()
        inputs[col] = [float(value)] if col in numeric_features else [value]
    input_df = pd.DataFrame(inputs)
    transformed_input = preprocessor.transform(input_df)

    prediction = model.predict(transformed_input)
    predicted_probability = prediction[0][0]
    result_label.config(text=f"Вероятность осложнений: {predicted_probability * 100:.2f}%")


# Создание GUI
root = tk.Tk()
root.title("Предсказание осложнений")

entries = {}
for i, col in enumerate(df.columns[:-1]):  # Убедитесь, что последний столбец 'Осложнения' не включен
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=5)
    label = tk.Label(frame, text=f"{col}:")
    label.pack(side='left')
    entry = tk.Entry(frame)
    entry.pack(side='right')
    entries[col] = entry

submit_button = tk.Button(root, text="Предсказать", command=submit)
submit_button.pack(pady=10)

result_label = tk.Label(root, text="Результат будет отображен здесь")
result_label.pack(pady=10)

root.mainloop()

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность (обучение)')
plt.plot(history.history['val_accuracy'], label='Точность (валидация)')
plt.title('Точность модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери (обучение)')
plt.plot(history.history['val_loss'], label='Потери (валидация)')
plt.title('Потери модели')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.show()
def input_new_data():
    # Запрашиваем данные от пользователя
    # Замените ниже 'input' на соответствующие запросы для каждого признака
    inputs = {}
    for col in df.columns[:-1]:  # исключаем последний столбец 'Осложнения'
        value = input(f"Введите значение для {col}: ")
        inputs[col] = [value]

    # Создаем DataFrame на основе введенных данных
    input_df = pd.DataFrame(inputs)

    # Преобразуем введенные данные
    transformed_input = preprocessor.transform(input_df)
    return transformed_input
#Предсказание на основе введенных данных
def predict_complications(transformed_data):
    prediction = model.predict(transformed_data)
    # Учитывая, что последний слой модели это сигмоид, выводим вероятность
    predicted_probability = prediction[0][0]
    print(f"Вероятность осложнений: {predicted_probability * 100:.2f}%")
# Обработка входных данных и выполнение предсказания
input_data = input_new_data()
predict_complications(input_data)



