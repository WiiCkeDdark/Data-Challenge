def find_crypto_df(crypto_data_list, crypto_name):
    for crypto_data in crypto_data_list:
        print(crypto_data["crypto"])
        if crypto_data["crypto"].lower() == crypto_name.lower():
            return crypto_data["dataFrame"]
    return None  # Return None if the crypto is not found
