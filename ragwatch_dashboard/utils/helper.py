def state_does_exist(state_name):
    import streamlit as st
    if state_name in st.session_state:
        return True
    return False

    
def state_get(state_name):
    import streamlit as st
    if state_does_exist(state_name):
        return st.session_state[state_name]
    return None
    

def read_csv_if_exists(file_path):
    import os
    import pandas as pd
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception:
            return None
    else:
        return None
    
def is_non_empty_df(df):
    return df is not None and not df.empty
