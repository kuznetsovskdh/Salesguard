import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

products = ['Ноутбук', 'Мышь', 'Клавиатура', 'Монитор', 'Наушники',
            'Веб-камера', 'USB-хаб', 'Кабель HDMI', 'Флешка', 'SSD']
customers = [f'C{i:03d}' for i in range(1, 51)]
regions = ['Москва', 'СПб', 'Екатеринбург', 'Новосибирск', 'Казань']

def make_rows(n=200):
    rows = []
    base = datetime(2023, 1, 1)
    for _ in range(n):
        qty = random.randint(1, 10)
        price = round(random.uniform(100, 5000), 2)
        rows.append({
            'date': base + timedelta(days=random.randint(0, 365)),
            'product': random.choice(products),
            'quantity': qty,
            'unit_price': price,
            'revenue': round(qty * price, 2),
            'customer_id': random.choice(customers),
            'region': random.choice(regions),
        })
    return rows

rows = make_rows(200)

# Намеренный мусор для теста нормализатора
rows[10]['revenue'] = '.'
rows[20]['revenue'] = None
rows[30]['date'] = 'banana'
rows[40]['revenue'] = -999

df = pd.DataFrame(rows)

# Файл 1: CSV с точкой запятой
df.to_csv('tests/test_semicolon.csv', sep=';', index=False, encoding='utf-8')

# Файл 2: XLSX
df.to_excel('tests/test_data.xlsx', index=False)

# Файл 3: JSON с другими именами колонок (имитация WB-выгрузки)
df_json = df.rename(columns={
    'date': 'order_dt',
    'revenue': 'retail_amount',
    'product': 'subject_name',
    'customer_id': 'customer'
})
df_json['order_dt'] = df_json['order_dt'].astype(str)
df_json.to_json('tests/test_wb_style.json', orient='records', force_ascii=False)

print('Готово: test_semicolon.csv, test_data.xlsx, test_wb_style.json')
