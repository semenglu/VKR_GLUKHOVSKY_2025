from data_loader import load_data
from preprocessing import build_preprocessor
from gui import launch_gui

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

# === 1. Загрузка и подготовка данных
df = load_data('kur5.KEE.csv', 'kur6.KAS.csv')

# Исключаем признаки, потенциально "подсказывающие" целевой результат
X = df.drop(columns=['Тип вмешательства', 'летальность', 'Осложнения', 'вид КЭЭ', 'ВПШ'])
y = df['Тип вмешательства']

# Выделяем категориальные и числовые признаки
categorical_features = X.select_dtypes(include='object').columns.tolist()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Убедимся, что все категориальные — строки
for col in categorical_features:
    X[col] = X[col].astype(str)

# === 2. Предобработка
preprocessor = build_preprocessor(numeric_features, categorical_features)
X_processed = preprocessor.fit_transform(X)

# Кодируем целевую переменную
label_encoder = LabelEncoder()
y_encoded = tf.keras.utils.to_categorical(label_encoder.fit_transform(y))

# Делим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42
)

# === 3. Построение улучшенной модели
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(2, activation='softmax')
])
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === 4. Обучение модели
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=16,
    validation_split=0.2
)

# === 5. Визуализация обучения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Обучение')
plt.plot(history.history['val_accuracy'], label='Валидация')
plt.title('Точность модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Обучение')
plt.plot(history.history['val_loss'], label='Валидация')
plt.title('Потери модели')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()

plt.tight_layout()
plt.show()

# === 6. Запуск GUI
launch_gui(preprocessor, model, label_encoder, numeric_features, categorical_features)
