import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats

# Custom CSS for dark mode
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTable {
        background-color: #2a2a2a;
        color: #f0f0f0;
    }
    .stMarkdown {
        color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

def create_step_by_step_solution(data, group_col, value_col):
    """Create detailed step-by-step Mann-Whitney U test solution"""
    
    st.header("📊 Mann-Whitney U Test: Step-by-Step Solution")
    
    # Step 1-3: Hypotheses and Significance Level
    st.write("### 1️⃣ Hypotheses")
    st.markdown(f"**H₀:** The median {value_col} in both groups are the same")
    st.markdown(f"**H₁:** The median {value_col} in both groups are not the same (Two-Tailed)")
    st.write("**Level of Significance (α):** 5% or 0.05")
    
    # Step 4: Combined Ranking
    st.write("### 2️⃣ Combined Ranking Process")
    
    # Create ranking table
    combined_data = []
    groups = sorted(data[group_col].unique())
    
    for _, row in data.iterrows():
        combined_data.append({
            'Group': row[group_col],
            'Value': row[value_col]
        })
    
    # Sort and rank with two decimal places
    df_ranked = pd.DataFrame(combined_data)
    df_ranked['Rank'] = np.round(stats.rankdata(df_ranked['Value']), 2)
    df_ranked = df_ranked.sort_values('Value')
    
    # Display ranking table with formatting
    st.write("**Combined Data with Ranks:**")
    styled_df = pd.DataFrame({
        'Group': df_ranked['Group'],
        'Value': df_ranked['Value'],
        'Rank': df_ranked['Rank']
    })
    
    st.table(styled_df.style.set_table_attributes('style="background-color: #2a2a2a; color: white;"').format({"Rank": "{:.2f}"}))
    
    # Calculate group sizes
    n1 = sum(df_ranked['Group'] == groups[0])
    n2 = sum(df_ranked['Group'] == groups[1])
    
    st.write(f"**n₁** (Number of observations in Group {groups[0]}) = {n1}")
    st.write(f"**n₂** (Number of observations in Group {groups[1]}) = {n2}")
    
    # Calculate rank sums
    R1 = df_ranked[df_ranked['Group'] == groups[0]]['Rank'].sum()
    R2 = df_ranked[df_ranked['Group'] == groups[1]]['Rank'].sum()
    
    st.write("\n### 3️⃣ Rank Sums Calculation")
    
    st.markdown(f"**R₁** (sum of ranks for Group {groups[0]}):")
    st.markdown(f"<p style='font-size: 18px; background-color: black; padding: 10px; border-radius: 5px;'><strong>= {' + '.join(map(str, df_ranked[df_ranked['Group'] == groups[0]]['Rank'].values))}</strong></p>", unsafe_allow_html=True)
    
    st.write(f"= {R1}")
    
    st.write(f"\n**R₂** (sum of ranks for Group {groups[1]}):")
    st.markdown(f"<p style='font-size: 18px; background-color: black; padding: 10px; border-radius: 5px;'><strong>= {' + '.join(map(str, df_ranked[df_ranked['Group'] == groups[1]]['Rank'].values))}</strong></p>", unsafe_allow_html=True)
    
    st.write(f"= {R2}")
    
    # Calculate U statistics
    st.write("\n### 4️⃣ Calculate U Statistics")
    
    U1_formula = f"""
    U₁ = n₁ × n₂ + (n₁(n₁ + 1)/2) - R₁
    = {n1} × {n2} + ({n1}({n1} + 1)/2) - {R1}
    = {n1 * n2} + ({n1 * (n1 + 1)}/2) - {R1}
    """
    
    U1 = n1 * n2 + (n1 * (n1 + 1))/2 - R1
    U1_formula += f"\n= {U1}"
    
    st.write("**U₁ Calculation:**")
    st.markdown(f"```math\n{U1_formula}\n```")
    
    U2_formula = f"""
    U₂ = n₁ × n₂ - U₁
    = {n1} × {n2} - {U1}
    = {n1 * n2} - {U1}
    """
    
    U2 = n1 * n2 - U1
    U2_formula += f"\n= {U2}"
    
    st.write("**U₂ Calculation:**")
    st.markdown(f"```math\n{U2_formula}\n```")
    
    # Test statistic
    U = min(U1, U2)
    
    st.write("\n### 5️⃣ Test Statistic")
    
    st.markdown(f"**U = min(U₁, U₂) = min({U1}, {U2}) = {U}**")
    
    # Critical value lookup
    st.write("\n### 6️⃣ Critical Value")
    st.write(f"Critical value (Uα) at 5% level of significance for n₁ = {n1} and n₂ = {n2}")
    
    # Comparison and Conclusion
    st.write("\n### 7️⃣ Comparison and Conclusion")
    
    p_value = stats.mannwhitneyu(
        df_ranked[df_ranked['Group'] == groups[0]]['Value'],
        df_ranked[df_ranked['Group'] == groups[1]]['Value'],
        alternative='two-sided'
    ).pvalue
    
    if p_value < 0.05:
        st.error(f"Since p-value ({p_value:.4f}) < α (0.05), we reject H₀")
        st.write(f"We conclude that the median {value_col} in the two groups are significantly different.")
    else:
        st.success(f"Since p-value ({p_value:.4f}) > α (0.05), we fail to reject H₀")
        st.write(f"We conclude that the median {value_col} in the two groups are not significantly different.")
    
    return {
        'n1': n1,
        'n2': n2,
        'R1': R1,
        'R2': R2,
        'U1': U1,
        'U2': U2,
        'U': U,
        'p_value': p_value
    }

def main():
   # Set page configuration at the very start

   # Application Title
   st.title("📐 Mann-Whitney U Test Calculator")
   st.write("Upload your data or use the sample data to perform a Mann-Whitney U test.")
   
   # Data input method selection
   input_method = st.radio(
       "Choose input method:",
       ["Upload CSV", "Use Sample Data"]
   )
   
   if input_method == "Upload CSV":
       file = st.file_uploader("Upload CSV file", type=["csv"])
       if file:
           try:
               data = pd.read_csv(file)
               st.write("Select columns for analysis:")
               group_col = st.selectbox("Select group column:", data.columns)
               value_col = st.selectbox("Select value column:", data.columns)
               
               if st.button("Perform Analysis"):
                   results = create_step_by_step_solution(data, group_col, value_col)
                   
           except Exception as e:
               st.error(f"Error: {str(e)}")
   else:
       # Sample data
       sample_data = pd.DataFrame({
           'Group': ['A', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'A'],
           'Value': [20, 23, 25, 29, 30, 35, 39, 42, 42, 51, 57, 60]
       })
       
       st.write("Sample Data:")
       st.dataframe(sample_data.style.set_table_attributes('style="background-color: #2a2a2a; color: black;"'))
       
       if st.button("Perform Analysis"):
           results = create_step_by_step_solution(sample_data, 'Group', 'Value')

if __name__ == "__main__":
   main()
