from data_loader import load_data
from preprocessing import build_preprocessor
from model import build_model
from gui import launch_gui
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Загрузка и подготовка данных
df = load_data('kur5.KEE.csv', 'kur6.KAS.csv')
X = df.drop(columns=['Тип вмешательства', 'летальность', 'Осложнения'])
y = df['Тип вмешательства']

categorical_features = X.select_dtypes(include='object').columns.tolist()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in categorical_features:
    X[col] = X[col].astype(str)

preprocessor = build_preprocessor(numeric_features, categorical_features)
X_processed = preprocessor.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = tf.keras.utils.to_categorical(label_encoder.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

model = build_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Запуск GUI
launch_gui(preprocessor, model, label_encoder, numeric_features, categorical_features)
