import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt

# === Загрузка и подготовка данных ===
df_kee = pd.read_csv('kur5.KEE.csv', sep=';')
df_kas = pd.read_csv('kur6.KAS.csv', sep=';')

df_kee.columns = ['Возраст', 'Пол', 'Степень стеноза внутренней сонной артерии', 'Стенокардия ФК', 'ПИМ',
                  'Нарушения ритма', 'ХСН', 'ФК по NYHA', 'СД', 'ХОБЛ', 'ОНМК в анамнезе', 'Вмешательство',
                  'вид КЭЭ', 'ВПШ', 'летальность', 'Осложнения']
df_kas.columns = ['Возраст', 'Пол', 'Степень стеноза внутренней сонной артерии', 'Стенокардия ФК', 'ПИМ',
                  'Нарушения ритма', 'ХСН', 'ФК по NYHA', 'СД', 'ХОБЛ', 'ОНМК в анамнезе', 'Вмешательство',
                  'летальность', 'Осложнения']

df_kee['Тип вмешательства'] = 'КЭЭ'
df_kas['Тип вмешательства'] = 'KAS'
df_kas['вид КЭЭ'] = 'отсутствует'
df_kas['ВПШ'] = 'отсутствует'

df_kee.drop(columns=['Вмешательство'], inplace=True)
df_kas.drop(columns=['Вмешательство'], inplace=True)

columns = ['Возраст', 'Пол', 'Степень стеноза внутренней сонной артерии', 'Стенокардия ФК', 'ПИМ',
           'Нарушения ритма', 'ХСН', 'ФК по NYHA', 'СД', 'ХОБЛ', 'ОНМК в анамнезе',
           'вид КЭЭ', 'ВПШ', 'летальность', 'Осложнения', 'Тип вмешательства']
df = pd.concat([df_kee[columns], df_kas[columns]], ignore_index=True)

X = df.drop(columns=['Тип вмешательства', 'летальность', 'Осложнения'])
y = df['Тип вмешательства']



# === Преобразование признаков ===
categorical_features = X.select_dtypes(include='object').columns.tolist()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in categorical_features:
    X[col] = X[col].astype(str)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_processed = preprocessor.fit_transform(X)

# === Кодирование целевой переменной ===
label_encoder = LabelEncoder()
y_encoded = tf.keras.utils.to_categorical(label_encoder.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# === Обучение модели ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# === Визуализация ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Обучение')
plt.plot(history.history['val_accuracy'], label='Валидация')
plt.title('Точность')
plt.xlabel('Эпоха')
plt.ylabel('Значение')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Обучение')
plt.plot(history.history['val_loss'], label='Валидация')
plt.title('Потери')
plt.xlabel('Эпоха')
plt.ylabel('Значение')
plt.legend()

plt.tight_layout()
plt.show()

# === GUI ===
def launch_gui():
    root = tk.Tk()
    root.title("Рекомендация типа вмешательства")
    root.geometry("600x850")
    entries = {}

    input_columns = [col for col in X.columns if col not in ['летальность', 'Осложнения']]
    predefined_values = {
        'Пол': ['М', 'Ж'],
        'ПИМ': ['0', '1'],
        'СД': ['0', '1'],
        'ХОБЛ': ['0', '1'],
        'ХСН': ['0', '1'],
        'ОНМК в анамнезе': ['0', '1', '2'],
        'ФК по NYHA': ['1', '2', '3', '4'],
        'Стенокардия ФК': ['0', '1', '2', '3'],
        'Нарушения ритма': ['0', '1'],
        'вид КЭЭ': ['отсутствует', '6', '11', '15', '16', '23'],
        'ВПШ': ['0', '1', 'отсутствует']
    }

    for col in input_columns:
        frame = tk.Frame(root)
        frame.pack(pady=5)
        label = tk.Label(frame, text=col + ":", width=30, anchor='w')
        label.pack(side='left')
        if col in predefined_values:
            var = tk.StringVar(value=predefined_values[col][0])
            menu = tk.OptionMenu(frame, var, *predefined_values[col])
            menu.pack(side='right')
            entries[col] = var
        else:
            entry = tk.Entry(frame, width=20)
            entry.pack(side='right')
            entries[col] = entry

    result_label = tk.Label(root, text="Результат появится здесь", font=("Arial", 14), fg="blue")
    result_label.pack(pady=20)

    def predict():
        try:
            input_data = {}
            for col in input_columns:
                widget = entries[col]
                val = widget.get()
                if col in numeric_features:
                    val = float(val)
                input_data[col] = [val]

            input_df = pd.DataFrame(input_data)
            for col in categorical_features:
                if col in input_df.columns:
                    input_df[col] = input_df[col].astype(str)

            transformed = preprocessor.transform(input_df)
            prediction = model.predict(transformed)
            predicted_class = prediction.argmax(axis=1)[0]
            result = label_encoder.inverse_transform([predicted_class])[0]
            result_label.config(text=f"Рекомендуется вмешательство: {result}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    tk.Button(root, text="Предсказать", command=predict, font=("Arial", 12), bg="lightgreen").pack(pady=10)
    root.mainloop()

launch_gui()
