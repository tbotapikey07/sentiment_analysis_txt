# pages/admin.py
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

def get_all_inputs():
    """Get all inputs from database"""
    conn = sqlite3.connect('user_inputs.db')
    df = pd.read_sql_query("SELECT * FROM inputs ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def delete_input(input_id):
    """Delete input by ID"""
    conn = sqlite3.connect('user_inputs.db')
    c = conn.cursor()
    c.execute("DELETE FROM inputs WHERE id=?", (input_id,))
    conn.commit()
    conn.close()

# App Configuration
st.set_page_config(page_title="Admin Panel", page_icon="🔐", layout="wide")

# Admin Page
st.title("🔐 Admin Panel")
st.write("View and manage all user submissions")
st.divider()

# Password protection
password = st.text_input("Enter Admin Password", type="password", key="admin_pw")

if password == "admin123":
    st.success("✅ Access Granted")
    st.divider()
    
    # Get data
    data = get_all_inputs()
    
    if not data.empty:
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Unique Users", data['user_name'].nunique())
        with col3:
            latest = data['timestamp'].max()
            st.metric("Latest Entry", latest[:10] if pd.notna(latest) else "N/A")
        
        st.divider()
        
        # Display table
        st.subheader("📊 All User Inputs")
        st.dataframe(data, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Delete section
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🗑️ Delete Record")
            record_id = st.number_input("Enter Record ID to delete", min_value=1, step=1)
        with col2:
            st.write("")
            st.write("")
            st.write("")
            if st.button("Delete Record", type="secondary", use_container_width=True):
                if record_id in data['id'].values:
                    delete_input(record_id)
                    st.success(f"Record {record_id} deleted successfully!")
                    st.rerun()
                else:
                    st.error("Record ID not found!")
        
        st.divider()
        
        # Download section
        st.subheader("📥 Export Data")
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download CSV File",
            data=csv,
            file_name=f"user_inputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("📭 No data available yet. Users haven't submitted any inputs.")
        
elif password:
    st.error("❌ Incorrect password. Please try again.")
else:
    st.info("🔒 Please enter the admin password to access this page")
    #st.caption("Default password: admin123")
