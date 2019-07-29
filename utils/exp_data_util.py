import numpy as np

def create_data(data_size=10000, max_transaction_history=50, max_product_click_history = 50, max_promotion_click_history = 50, category_size = 100, numeric_size = 10):

    data1 = np.random.randint(category_size, size=(data_size, max_transaction_history))
    data2 = np.random.randint(category_size, size=(data_size, max_product_click_history))
    data3 = np.random.randint(category_size, size=(data_size, max_promotion_click_history))
    inputs = [data1, data2, data3]

    single_category_cols = {105:3,106:5,107:10}   ## such as location : unique_value_size
    for k in single_category_cols:
        inputs.append(np.random.randint(single_category_cols[k], size=(data_size, 1)))

    num1 = np.random.random(size=(data_size, numeric_size))
    inputs.append(num1)

    labels = np.random.randint(category_size, size=(data_size, 1))

    return inputs, labels