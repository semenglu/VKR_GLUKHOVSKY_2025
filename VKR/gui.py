import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import sqlite3
from utils import stenosis_category
from database import save_to_db

class InterventionApp:
    def __init__(self, root, preprocessor, model, label_encoder, numeric_features, categorical_features):
        self.root = root
        self.preprocessor = preprocessor
        self.model = model
        self.label_encoder = label_encoder
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.input_data = {}
        self.last_prediction = None

        self.input_columns = numeric_features + categorical_features
        self.input_columns = [col for col in self.input_columns if col not in ['летальность', 'Осложнения']]
        self.input_columns = list(dict.fromkeys(self.input_columns))  # remove duplicates

        self.predefined_values = {
            'Пол': ['М', 'Ж'],
            'ПИМ': ['0', '1'],
            'СД': ['0', '1'],
            'ХОБЛ': ['0', '1', '2', '3'],
            'ХСН': ['0', '1'],
            'ОНМК в анамнезе': ['0', '1', '2'],
            'ФК по NYHA': ['1', '2', '3', '4'],
            'Стенокардия ФК': ['0', '1', '2', '3', '4'],
            'Нарушения ритма': ['0', '1'],
            'вид КЭЭ': ['отсутствует', '6', '11', '15', '16', '23'],
            'ВПШ': ['0', '1', 'отсутствует']
        }

        self.entries = {}
        self.build_interface()

    def build_interface(self):
        self.root.title("Рекомендация типа вмешательства")
        self.root.configure(bg='white')
        self.root.geometry("600x900")

        tk.Label(self.root, text="Введите данные пациента", font=("Arial", 16, "bold"), bg='white').pack(pady=(15, 10))
        form_frame = tk.Frame(self.root, bg='white')
        form_frame.pack(pady=10)

        for i, col in enumerate(self.input_columns):
            tk.Label(form_frame, text=col + ":", font=("Arial", 11), bg='white', anchor='w', width=35).grid(row=i, column=0, sticky='w', padx=5, pady=4)
            if col == 'Степень стеноза внутренней сонной артерии':
                entry = tk.Entry(form_frame, width=18)
                entry.grid(row=i, column=1, padx=5)
                self.entries[col] = entry
            elif col in self.predefined_values:
                var = tk.StringVar(value=self.predefined_values[col][0])
                tk.OptionMenu(form_frame, var, *self.predefined_values[col]).grid(row=i, column=1, padx=5)
                self.entries[col] = var
            else:
                entry = tk.Entry(form_frame, width=18)
                entry.grid(row=i, column=1, padx=5)
                self.entries[col] = entry

        self.result_label = tk.Label(self.root, text="Результат появится здесь", font=("Arial", 14), fg="#0033cc", bg='white')
        self.result_label.pack(pady=20)

        tk.Button(self.root, text="Предсказать", command=self.predict, font=("Arial", 12, "bold"), bg="#a8f0a5", width=20, height=2).pack(pady=10)
        tk.Button(self.root, text="Сохранить в БД", command=self.save_result, font=("Arial", 10), bg="#f0e68c", width=20).pack(pady=5)
        tk.Button(self.root, text="Показать сохраненные", command=self.show_records, font=("Arial", 10), bg="#add8e6", width=20).pack(pady=5)

    def predict(self):
        try:
            input_dict = {}
            for col in self.input_columns:
                val = self.entries[col].get()
                if col == 'Степень стеноза внутренней сонной артерии':
                    val = stenosis_category(val)
                if col in self.numeric_features and col != 'Степень стеноза внутренней сонной артерии':
                    val = float(val)
                input_dict[col] = [val]

            input_df = pd.DataFrame(input_dict)
            for col in self.categorical_features:
                if col in input_df.columns:
                    input_df[col] = input_df[col].astype(str)

            transformed = self.preprocessor.transform(input_df)
            prediction = self.model.predict(transformed)
            predicted_class = prediction.argmax(axis=1)[0]
            result = self.label_encoder.inverse_transform([predicted_class])[0]

            self.result_label.config(text=f"Рекомендуется вмешательство: {result}", fg="#007f00")
            self.input_data = input_dict
            self.last_prediction = result
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def save_result(self):
        if not self.input_data or not self.last_prediction:
            messagebox.showwarning("Внимание", "Сначала выполните предсказание")
            return
        save_to_db(self.input_data, self.last_prediction)
        messagebox.showinfo("Успешно", "Данные сохранены в базу")

    def show_records(self):
        window = tk.Toplevel(self.root)
        window.title("Сохраненные предсказания")
        window.geometry("1200x400")
        tree = ttk.Treeview(window)

        conn = sqlite3.connect('interventions.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions")
        rows = cursor.fetchall()
        conn.close()

        columns = [
            'ID', 'Возраст', 'Пол', 'Стеноз', 'Стенокардия', 'ПИМ', 'Ритм', 'ХСН',
            'NYHA', 'СД', 'ХОБЛ', 'ОНМК', 'вид КЭЭ', 'ВПШ', 'Тип вмешательства'
        ]
        tree['columns'] = columns
        tree['show'] = 'headings'
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')
        for row in rows:
            tree.insert("", "end", values=row)
        tree.pack(expand=True, fill='both')

def launch_gui(preprocessor, model, label_encoder, numeric_features, categorical_features):
    root = tk.Tk()
    app = InterventionApp(root, preprocessor, model, label_encoder, numeric_features, categorical_features)
    root.mainloop()
