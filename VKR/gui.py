import tkinter as tk
from tkinter import messagebox
import pandas as pd
from utils import stenosis_category

def launch_gui(preprocessor, model, label_encoder, numeric_features, categorical_features):
    root = tk.Tk()
    root.title("Рекомендация типа вмешательства")
    root.configure(bg='white')
    root.geometry("600x850")
    entries = {}

    input_columns = numeric_features + categorical_features
    input_columns = [col for col in input_columns if col not in ['летальность', 'Осложнения']]
    input_columns = list(dict.fromkeys(input_columns))  # remove duplicates

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

    def predict():
        try:
            input_data = {}
            for col in input_columns:
                val = entries[col].get()
                if col == 'Степень стеноза внутренней сонной артерии':
                    val = stenosis_category(val)
                if col in numeric_features and col != 'Степень стеноза внутренней сонной артерии':
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
            result_label.config(text=f"Рекомендуется вмешательство: {result}", fg="#007f00")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    tk.Label(root, text="Введите данные пациента", font=("Arial", 16, "bold"), bg='white').pack(pady=(15, 10))
    form_frame = tk.Frame(root, bg='white')
    form_frame.pack(pady=10)

    for i, col in enumerate(input_columns):
        tk.Label(form_frame, text=col + ":", font=("Arial", 11), bg='white', anchor='w', width=35).grid(row=i, column=0, sticky='w', padx=5, pady=4)
        if col == 'Степень стеноза внутренней сонной артерии':
            entry = tk.Entry(form_frame, width=18)
            entry.grid(row=i, column=1, padx=5)
            entries[col] = entry
        elif col in predefined_values:
            var = tk.StringVar(value=predefined_values[col][0])
            tk.OptionMenu(form_frame, var, *predefined_values[col]).grid(row=i, column=1, padx=5)
            entries[col] = var
        else:
            entry = tk.Entry(form_frame, width=18)
            entry.grid(row=i, column=1, padx=5)
            entries[col] = entry

    result_label = tk.Label(root, text="Результат появится здесь", font=("Arial", 14), fg="#0033cc", bg='white')
    result_label.pack(pady=20)
    tk.Button(root, text="Предсказать", command=predict, font=("Arial", 12, "bold"), bg="#a8f0a5", width=20, height=2).pack(pady=10)
    root.mainloop()
