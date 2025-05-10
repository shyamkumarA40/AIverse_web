import streamlit as st
import pandas as pd
import sqlite3
import os

st.set_page_config(page_title="AIverse CSV to DB", layout="centered")
st.title("📄 CSV to Database App")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset:")
    st.dataframe(df)

    # Save CSV to DB
    db_path = "uploaded_data.db"
    table_name = "UserData"

    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    # Get columns from uploaded data
    features = df.columns.tolist()
    st.write("### Input New Data")

    # Allow user to input data for up to 4 columns
    input_data = {}
    for i, col in enumerate(features[:21]):
        input_data[col] = st.text_input(f"Enter value for {col}", key=col)

    if st.button("Submit New Row"):
        # Fill missing inputs with None or empty string
        new_row = [input_data.get(col, None) for col in df.columns]

        # Insert the row into the database
        placeholders = ",".join("?" * len(new_row))
        insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"
        conn.execute(insert_query, new_row)
        conn.commit()
        st.success("New row added to the database.")

    # Display updated data
    updated_df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    st.write("### Updated Data:")
    st.dataframe(updated_df)

    # Option to download updated CSV
    csv = updated_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "updated_data.csv", "text/csv")

    conn.close()
else:
    st.info("Please upload a CSV file to begin.")

