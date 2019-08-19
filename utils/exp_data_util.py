import numpy as np

def create_data(data_size=10000, max_transaction_history=20, max_product_click_history = 20, max_promotion_click_history = 20, category_size = 100, numeric_size = 1):

    data1 = np.random.randint(category_size, size=(data_size, max_transaction_history-1))
    data2 = np.random.randint(category_size, size=(data_size, max_promotion_click_history-2))
    inputs = [data1, data2]

    single_category_cols = {'genderIndex':(3, 8),'is_email_verifiedIndex': (2, 8), 'age':(12, 8), 'cityIndex':(922,16)}   ## such as location : unique_value_size, embedding_size
    for k in single_category_cols:
        inputs.append(np.random.randint(single_category_cols[k][0], size=(data_size, 1)))

    labels = np.random.randint(category_size, size=(data_size, 1))

    return inputs, labels