import streamlit as st
from ge_matcher import get_competitor_data, find_best_ge_match, GE_PRODUCTS

st.title("GE Appliances Matcher")

sku = st.text_input("Enter competitor SKU:")
if sku:
    comp = get_competitor_data(sku)
    if comp.category == "unknown":
        st.error(f"No data available for SKU {sku}")
    else:
        match = find_best_ge_match(comp, GE_PRODUCTS)
        if match:
            st.success(f"Best GE match for {sku} is {match.sku}")
            st.write("Competitor details:")
            st.json(comp.__dict__)
            st.write("Matched GE product details:")
            st.json(match.__dict__)
        else:
            st.warning("No matching GE product found.")
