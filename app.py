import os
import io
import pandas as pd
import streamlit as st
import snowflake.connector
import numpy as np
import time
import requests
from gpt import OpenAIService
import base64

# Function to get the base64 encoded version of the image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Using the function to get base64 string for 'omnia-logo.png'
img_file = 'omnia-logo.png'
img_base64 = get_base64_of_bin_file(img_file)

# Set wide layout for Streamlit app
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown(
    f"""
    <style>
        body {{
            background-color: #00008B; /* Dark blue background */
            background-image: url("data:omnia-logo.png;base64,{img_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: right top;
        }}
        .sidebar .sidebar-content {{
            background-color: #00008B; /* Sidebar background */
        }}
        .stTextInput, .stTextArea {{
            background-color: #D3D3D3; /* Input fields */
            color: black;
            border-radius: 15px;
            border: 2px solid #00bcd4;
            margin: 10px 0;
            padding: 10px;
        }}
        .stButton button {{
            background-color: #00bcd4; /* Button background */
            color: #ffffff;
            border: 2px solid #0097a7;
            border-radius: 15px;
            margin: 10px 0;
        }}
        .stButton button:hover {{
            background-color: #0097a7; /* Button hover */
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #006064; /* Heading colors */
        }}
        [title="Show password text"] {{
            display: none; /* Hide password text */
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Function to read the prompt file content
def read_prompt_file(fname):
    with open(fname, "r", encoding='utf-8') as f:
        return f.read()

# Caching the Snowflake connection
@st.cache_resource
class SnowflakeDB:
    def __init__(self) -> None:
        self.conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA")
        )
        self.cursor = self.conn.cursor()

@st.cache_data
def query(_conn, query_text):
    try:
        return pd.read_sql(query_text, _conn)
    except Exception as e:
        st.warning("Error in query")
        st.error(e)
        return None

@st.cache_data
def ask(prompt):
    gpt = OpenAIService()
    response = gpt.prompt(prompt)
    return response["choices"][0]["message"]["content"] if response else None

def get_tables_schema(_conn):
    with st.expander("View tables schema in database"):
        table_schemas = ""
        df = query(_conn, "SHOW TABLES")
        st.write(df)
        for table in df["name"]:
            t = f"{os.getenv('SNOWFLAKE_DATABASE')}.{os.getenv('SNOWFLAKE_SCHEMA')}.{table}"
            ddl_query = f"SELECT GET_DDL('table', '{t}');"
            ddl = query(_conn, ddl_query)
            schema = f"\n{ddl.iloc[0, 0]}\n" if ddl is not None and not ddl.empty else "No schema available."
            st.write(f"### {table}")
            st.markdown(f"```sql\n{schema}\n```")
            st.write("---")
            table_schemas += f"\n{table}\n{schema}\n"
    return table_schemas

def sanitize_input(input_string):
    return input_string.encode('utf-8', 'ignore').decode('utf-8')

def validate_sql(sql):
    restricted_keywords = ["DROP", "ALTER", "TRUNCATE", "UPDATE", "REMOVE"]
    for keyword in restricted_keywords:
        if keyword in sql.upper():
            return False, keyword
    return True, None

def df_schema(df):
    sio = io.StringIO()
    df.info(buf=sio)
    df_info = sio.getvalue()
    return df_info

def main():
    if 'openai_key_entered' not in st.session_state:
        st.session_state.openai_key_entered = False
    if 'question' not in st.session_state:
        st.session_state.question = ""

    api_key = None

    with st.sidebar:
        st.image("omnia-logo.png", use_column_width=True)
        st.title("Please fill the details: ")
        option = st.selectbox("Choose an option", ["Snowflake Credentials", "OpenAI API Key"], index=1)

        if option == "Snowflake Credentials":
            st.caption("Snowflake Credentials")
            os.environ["SNOWFLAKE_USER"] = st.text_input("User", value="Siri")
            os.environ["SNOWFLAKE_PASSWORD"] = st.text_input("Password", value="Techment@123", key="hidden_password", type="password")
            os.environ["SNOWFLAKE_ACCOUNT"] = st.text_input("Account", value="jv51685.central-india.azure")
            os.environ["SNOWFLAKE_WAREHOUSE"] = st.text_input("Warehouse", value="COMPUTE_WH")
            os.environ["SNOWFLAKE_SCHEMA"] = st.text_input("Schema", value="RAW")
            os.environ["SNOWFLAKE_DATABASE"] = st.text_input("Database", value="SALES_REVENUE_DATA")
            st.warning("Enter OpenAI API Key to proceed.")
            st.session_state.openai_key_entered = False

        elif option == "OpenAI API Key":
            api_key = st.text_input("OpenAI API Key", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("OpenAI API Key entered successfully.")
                st.session_state.openai_key_entered = True

        if api_key or st.session_state.openai_key_entered:
            st.session_state.question = st.text_area(
                "Hi! I’m Snow Chat, your data assistant. Ask me anything about your connected data!",
                value=st.session_state.question,
                placeholder="What is the total revenue?",
                height=100,
            )

    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please provide a valid OpenAI API key.")
        st.stop()

    if st.session_state.openai_key_entered:
        if all(env_var in os.environ for env_var in ["SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]):
            sf = SnowflakeDB()
        else:
            st.stop()
    else:
        st.warning("Please enter the OpenAI API Key to proceed.")
        st.stop()

    question = st.session_state.question.strip()

    if question:
        st.title("Snowflake-Streamlit Integration")
        st.write("Connect to your Snowflake database and ask questions about your data in real-time.")
        
        table_schemas = get_tables_schema(sf.conn)

        prompt = read_prompt_file("sql_prompt.txt")
        prompt = prompt.replace("<<TABLES>>", table_schemas)
        prompt = prompt.replace("<<QUESTION>>", question)

        answer = ask(prompt)

        if answer is None:
            st.error("Failed to get a response from OpenAI.")
            return

        answer = answer.replace("```sql", "").replace("```", "").strip()

        st.write("Generated SQL Query:")
        st.code(answer)

        is_valid, keyword = validate_sql(answer)
        if not is_valid:
            st.error(f"Operation not allowed: {keyword} is restricted!")
            st.stop()
        
        df = query(sf.conn, answer)
        
        if df is not None:
            st.write("Query Result")
            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.error("The query returned an empty DataFrame.")
        else:
            st.error("Query execution failed.")

        python_question = st.text_input("Ask a question about the result", placeholder="e.g. Visualize the data")
        if python_question:
            df_info = str(df.head())  

            prompt = read_prompt_file("python_prompt.txt")
            prompt = prompt.replace("<<DATAFRAME>>", df_info)
            prompt = prompt.replace("<<QUESTION>>", python_question)
            code_answer = ask(prompt).replace("```python", "").replace("```", "").strip()
            with st.expander("View generated code"):
                st.code(code_answer)
            exec(code_answer)

if __name__ == "__main__":
    main()

