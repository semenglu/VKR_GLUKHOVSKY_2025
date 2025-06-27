
import sqlite3

def init_db():
    conn = sqlite3.connect('interventions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            gender TEXT,
            stenosis TEXT,
            fk TEXT,
            pim TEXT,
            rhythm TEXT,
            hsn TEXT,
            nyha TEXT,
            diabetes TEXT,
            hobl TEXT,
            onmk TEXT,
            kee_type TEXT,
            vpsh TEXT,
            prediction TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(data: dict, prediction: str):
    conn = sqlite3.connect('interventions.db')
    cursor = conn.cursor()
    values = (
        int(data.get('Возраст', [0])[0]),
        data.get('Пол', [''])[0],
        data.get('Степень стеноза внутренней сонной артерии', [''])[0],
        data.get('Стенокардия ФК', [''])[0],
        data.get('ПИМ', [''])[0],
        data.get('Нарушения ритма', [''])[0],
        data.get('ХСН', [''])[0],
        data.get('ФК по NYHA', [''])[0],
        data.get('СД', [''])[0],
        data.get('ХОБЛ', [''])[0],
        data.get('ОНМК в анамнезе', [''])[0],
        data.get('вид КЭЭ', [''])[0],
        data.get('ВПШ', [''])[0],
        prediction
    )
    cursor.execute('''
        INSERT INTO predictions (
            age, gender, stenosis, angina, pim, rhythm, chf, nyha,
            diabetes, copd, stroke, kee_type, vpsh, prediction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', values)
    conn.commit()
    conn.close()
