from data_loader import load_data
from preprocessing import build_preprocessor
from gui import launch_gui

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Загрузка и подготовка данных
df = load_data('kur5.KEE.csv', 'kur6.KAS.csv')
X = df.drop(columns=['Тип вмешательства', 'летальность', 'Осложнения', 'вид КЭЭ', 'ВПШ'])
y = df['Тип вмешательства']

categorical_features = X.select_dtypes(include='object').columns.tolist()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in categorical_features:
    X[col] = X[col].astype(str)

preprocessor = build_preprocessor(numeric_features, categorical_features)
X_processed = preprocessor.fit_transform(X)

label_encoder = LabelEncoder()
y_labels = label_encoder.fit_transform(y)
y_encoded = tf.keras.utils.to_categorical(y_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42
)

#Вычисление весов классов
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_labels),
    y=y_labels
)
class_weight_dict = dict(zip(np.unique(y_labels), class_weights))

#Построение улучшенной модели
from model import build_model
model = build_model(X_train.shape[1])

# Обучение модели с EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)

# Оценка модели
y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)
y_true = y_test.argmax(axis=1)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))

# Визуализация обучения и матрицы ошибок
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

# Матрица ошибок
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Факт')
    plt.xlabel('Прогноз')
    plt.title('Матрица ошибок')
    plt.tight_layout()
    plt.show()

cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, label_encoder.classes_)

from database import init_db
init_db()

# Запуск GUI
launch_gui(preprocessor, model, label_encoder, numeric_features, categorical_features)