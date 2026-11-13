import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Retirement Home Location Analysis - Brazil v2",
    page_icon="🏠", 
    layout="wide"
)

# Cache data loading functions
@st.cache_data
def load_data_with_encoding(file_path, sep=';'):
    """Load CSV with multiple encoding attempts"""
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            df = pd.read_csv(file_path, sep=sep, encoding=encoding, dtype=str)
            df.columns = df.columns.str.strip('"')
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.strip('"')
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {file_path}")

@st.cache_data
def prepare_municipality_data():
    """Load and prepare municipality-level data using correct V01041 aggregation"""
    
    # STEP 1: Load district data (has V01041 - elderly 70+ population)
    district_df = load_data_with_encoding("Dados IBGE/Distrito/Agregados_por_distritos_demografia_BR.csv")
    
    # STEP 2: Extract municipality codes from district codes and aggregate V01041
    district_df['CD_MUN_extracted'] = district_df['CD_DIST'].str[:7]
    district_df['V01041_num'] = pd.to_numeric(district_df['V01041'], errors='coerce').fillna(0)
    
    # Aggregate elderly population (V01041) by municipality
    mun_elderly = district_df.groupby('CD_MUN_extracted')['V01041_num'].sum().reset_index()
    mun_elderly.columns = ['CD_MUN', 'elderly_total']
    
    # STEP 3: Load municipality income data
    income_df = load_data_with_encoding("Dados IBGE/Municipio/renda media municipio.csv")
    
    # STEP 4: Load municipality info (has names, states, and total population)
    info_df = load_data_with_encoding("Dados IBGE/Municipio/infos sobre municipios.csv")
    
    # STEP 5: Get total population from municipality demographics file
    demo_df = load_data_with_encoding("Dados IBGE/Municipio/demografia municipios.csv")
    
    # Use V00005 as total population (verified to match census data)
    demo_df['total_population'] = pd.to_numeric(demo_df.get('V00005', 0), errors='coerce').fillna(0)
    total_pop_data = demo_df[['CD_MUN', 'total_population']]
    
    # STEP 6: Clean income data - municipality file has different column names
    income_cols_to_use = ['CD_MUN']
    
    if 'Rendaa media do responsavel' in income_df.columns:
        income_df['avg_income_raw'] = pd.to_numeric(
            income_df['Rendaa media do responsavel'].astype(str).str.replace(',', '.'), 
            errors='coerce'
        )
        income_cols_to_use.append('avg_income_raw')
    elif 'V06004' in income_df.columns:
        income_df['avg_income_raw'] = pd.to_numeric(income_df['V06004'], errors='coerce')
        income_cols_to_use.append('avg_income_raw')
    
    if 'Variancia' in income_df.columns:
        income_df['income_variance_raw'] = pd.to_numeric(
            income_df['Variancia'].astype(str).str.replace(',', '.'), 
            errors='coerce'
        )
        income_cols_to_use.append('income_variance_raw')
    elif 'V06005' in income_df.columns:
        income_df['income_variance_raw'] = pd.to_numeric(income_df['V06005'], errors='coerce')
        income_cols_to_use.append('income_variance_raw')
    
    # STEP 7: Merge all data sources
    # Start with elderly data (aggregated from districts using V01041)
    combined = mun_elderly.merge(
        info_df[['CD_MUN', 'NM_MUN', 'NM_UF']], 
        on='CD_MUN', how='left'
    ).merge(
        total_pop_data,
        on='CD_MUN', how='left'
    ).merge(
        income_df[income_cols_to_use], 
        on='CD_MUN', how='left'
    )
    
    # STEP 8: Clean and prepare final columns
    combined['avg_income'] = combined.get('avg_income_raw', 0)
    combined['income_variance'] = combined.get('income_variance_raw', 0)
    combined['income_std'] = np.sqrt(combined['income_variance'].fillna(0))
    combined['name'] = combined['NM_MUN']
    combined['state'] = combined['NM_UF']
    
    # Calculate elderly percentage for verification
    combined['elderly_percentage'] = np.where(
        combined['total_population'] > 0,
        (combined['elderly_total'] / combined['total_population'] * 100).round(1),
        0
    )
    
    # Add data source note for transparency
    combined['data_source_note'] = 'Elderly 70+ from V01041 (district aggregation)'
    
    return combined

@st.cache_data
def load_hierarchy_data():
    """Load district and neighborhood data for hierarchical navigation"""
    
    # Load district data
    district_df = load_data_with_encoding("Dados IBGE/Distrito/Agregados_por_distritos_demografia_BR.csv")
    district_income_df = load_data_with_encoding("Dados IBGE/Distrito/Agregados_por_distritos_renda_responsavel_BR.csv")
    
    # Load neighborhood data  
    neighborhood_df = load_data_with_encoding("Dados IBGE/Bairro/Agregados_por_bairros_demografia_BR.csv")
    neighborhood_income_df = load_data_with_encoding("Dados IBGE/Bairro/Agregados_por_bairros_renda_responsavel_BR.csv")
    
    # Load hierarchy mapping
    hierarchy_df = load_data_with_encoding("Dados IBGE/Bairro/Info sobre os bairros.csv")
    
    # Prepare district data
    district_data = district_df.copy()
    district_data['elderly_total'] = pd.to_numeric(district_data['V01041'], errors='coerce').fillna(0)
    district_data['CD_MUN'] = district_data['CD_DIST'].str[:7]  # Extract municipality code
    
    # Merge with income data
    district_data = district_data.merge(
        district_income_df[['CD_DIST', 'V06004', 'V06005']], 
        on='CD_DIST', how='left'
    )
    
    # Convert income data (handle comma decimal separator)
    district_data['avg_income'] = pd.to_numeric(
        district_data['V06004'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    ).fillna(0)
    
    district_data['income_variance'] = pd.to_numeric(
        district_data['V06005'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    ).fillna(0)
    
    district_data['income_std'] = np.sqrt(district_data['income_variance'])
    
    # Prepare neighborhood data
    neighborhood_data = neighborhood_df.copy()
    neighborhood_data['elderly_total'] = pd.to_numeric(neighborhood_data['V01041'], errors='coerce').fillna(0)
    
    # Merge with income data
    neighborhood_data = neighborhood_data.merge(
        neighborhood_income_df[['CD_BAIRRO', 'V06004', 'V06005']], 
        on='CD_BAIRRO', how='left'
    )
    
    # Convert income data (handle comma decimal separator)
    neighborhood_data['avg_income'] = pd.to_numeric(
        neighborhood_data['V06004'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    ).fillna(0)
    
    neighborhood_data['income_variance'] = pd.to_numeric(
        neighborhood_data['V06005'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    ).fillna(0)
    
    neighborhood_data['income_std'] = np.sqrt(neighborhood_data['income_variance'])
    
    # Merge with hierarchy to get full geographic information
    hierarchy_cols = ['CD_BAIRRO', 'CD_DIST', 'NM_DIST', 'CD_MUN', 'NM_MUN', 'NM_UF']
    # Only include columns that exist in hierarchy
    available_hierarchy_cols = [col for col in hierarchy_cols if col in hierarchy_df.columns]
    
    if available_hierarchy_cols:
        neighborhood_data = neighborhood_data.merge(
            hierarchy_df[available_hierarchy_cols],
            on='CD_BAIRRO', how='left'
        )
    
    return {
        'districts': district_data,
        'neighborhoods': neighborhood_data,
        'hierarchy': hierarchy_df
    }

def calculate_elders_above_8k(elderly_pop, avg_income, income_std, limite=8000):
    """
    Calculate number of elders making above R$ 8,000 using log-normal distribution
    
    Parameters:
    - elderly_pop: total elderly population (70+)
    - avg_income: average income of household heads
    - income_std: standard deviation of income
    - limite: income threshold (default: 8000)
    
    Returns:
    - number of elders estimated to earn above the threshold
    
    Note: IBGE variance data often produces unrealistic CV values (>200%).
    We cap CV at 1.0 (100%) for reasonable income distribution modeling.
    """
    if pd.isna(elderly_pop) or pd.isna(avg_income) or pd.isna(income_std):
        return 0
    
    if elderly_pop == 0 or avg_income <= 0 or income_std <= 0:
        return 0
    
    try:
        # Calculate coefficient of variation
        cv = income_std / avg_income
        
        # Cap CV at reasonable maximum (IBGE data often has unrealistic variance)
        # Typical income CV ranges: 0.3-0.8 (normal), 0.8-1.2 (high inequality)
        # CV > 2.0 suggests data quality issues - we cap at 1.0 for reasonable modeling
        if cv > 1.0:
            cv = 1.0
        
        # Parameters of log-normal distribution
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(avg_income) - (sigma**2 / 2)
        
        # Z-score in log space
        z_score = (np.log(limite) - mu) / sigma
        
        # Probability using standard normal CDF (approximation)
        # Using error function approximation since we don't have scipy
        def erf_approx(x):
            # Abramowitz and Stegun approximation
            a1 =  0.254829592
            a2 = -0.284496736
            a3 =  1.421413741
            a4 = -1.453152027
            a5 =  1.061405429
            p  =  0.3275911
            
            sign = 1 if x >= 0 else -1
            x = abs(x)
            
            t = 1.0 / (1.0 + p * x)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
            
            return sign * y
        
        def normal_cdf(z):
            return 0.5 * (1 + erf_approx(z / np.sqrt(2)))
        
        # Probability of being below threshold
        prob_abaixo = normal_cdf(z_score)
        
        # Percentage above threshold
        percentual_acima = (1 - prob_abaixo) * 100
        
        # Number of elders above threshold
        elders_above_8k = (percentual_acima / 100) * elderly_pop
        
        return round(elders_above_8k, 0)
        
    except Exception:
        # Return 0 if calculation fails
        return 0

def main():
    st.title("🏠 Retirement Home Location Analysis - Brazil v2")
    st.markdown("**Enhanced version using correct V01041 elderly population data**")
    
    # Data Source and Elders Above R$ 8k Explanation
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.expander("📊 Data Sources & Methodology"):
            st.markdown("""
            **🧓 Elderly Population (70+):**
            - Source: IBGE V01041 code (official elderly 70+ count)
            - Method: Aggregated from district-level data to municipality level
            - Verified against Census 2022 population data
            
            **💰 Income Data:**
            - Average income of household heads
            - Income variance for stability analysis
            
            **🏛️ Geographic Coverage:**
            - All 5,570 Brazilian municipalities
            - Data from IBGE Census 2022
            """)
    
    with col2:
        with st.expander("ℹ️ Elders Above R$ 8k Calculation"):
            st.markdown("""
            **📊 Log-Normal Distribution Method**
            - Uses household head income data to estimate elderly purchasing power
            - Assumes income follows log-normal distribution (typical for income data)
            - Calculates number of elders earning above R$ 8,000 threshold
            
            **🧮 Calculation Steps:**
            1. **Coefficient of Variation** = Income Std Dev ÷ Average Income
            2. **Log-Normal Parameters** calculated from mean and std dev
            3. **Z-Score** computed for R$ 8,000 threshold
            4. **Probability** of earning above threshold using normal CDF
            5. **Final Count** = Elderly Population × Probability
            
            **💡 Interpretation:**
            - Higher values indicate more affluent elderly population
            - Represents estimated market size for premium retirement services
            - Based on R$ 8,000+ monthly income (Brazilian Class B+)
            """)
        
    st.markdown("---")
    
    # Load data
    try:
        with st.spinner("Loading data..."):
            municipality_data = prepare_municipality_data()
            hierarchy_data = load_hierarchy_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return
    
    # Create tabs for different features
    tab1, tab2 = st.tabs(["🗺️ Geographic Analysis", "⚖️ Side-by-Side Comparison"])
    
    with tab1:
        # Hierarchical Navigation Interface
        st.header("🗺️ Geographic Navigation")
    
        # Create columns for dropdowns
        col1, col2, col3, col4 = st.columns(4)
        
        # Initialize session state
        if 'selected_state' not in st.session_state:
            st.session_state.selected_state = None
        if 'selected_city' not in st.session_state:
            st.session_state.selected_city = None  
        if 'selected_district' not in st.session_state:
            st.session_state.selected_district = None
        if 'selected_neighborhood' not in st.session_state:
            st.session_state.selected_neighborhood = None
        
        # State Dropdown
        with col1:
            st.subheader("🏛️ State")
            available_states = sorted(municipality_data['state'].dropna().unique())
            
            selected_state = st.selectbox(
                "Choose a state:",
                options=[None] + available_states,
                format_func=lambda x: "Select state..." if x is None else x,
                key="state_selector"
            )
            
            # Reset downstream selections when state changes
            if selected_state != st.session_state.selected_state:
                st.session_state.selected_state = selected_state
                st.session_state.selected_city = None
                st.session_state.selected_district = None
                st.session_state.selected_neighborhood = None
        
        # City Dropdown (only show if state selected)
        with col2:
            st.subheader("🏙️ Municipality")
            if selected_state:
                available_cities = municipality_data[
                    municipality_data['state'] == selected_state
                ]['name'].dropna().sort_values().unique()
                
                selected_city = st.selectbox(
                    "Choose a municipality:",
                    options=[None] + list(available_cities),
                    format_func=lambda x: "Select municipality..." if x is None else x,
                    key="city_selector"
                )
                
                # Reset downstream selections when city changes
                if selected_city != st.session_state.selected_city:
                    st.session_state.selected_city = selected_city
                    st.session_state.selected_district = None
                    st.session_state.selected_neighborhood = None
            else:
                st.info("Select a state first")
                selected_city = None
        
        # District Dropdown (only show if city selected and has districts)
        with col3:
            st.subheader("🏘️ District")
            if selected_city:
                # Get municipality code to find its districts
                city_info = municipality_data[
                    (municipality_data['state'] == selected_state) & 
                    (municipality_data['name'] == selected_city)
                ]
                
                if len(city_info) > 0:
                    mun_code = city_info['CD_MUN'].iloc[0]
                    
                    # Find districts for this municipality
                    available_districts = hierarchy_data['districts'][
                        hierarchy_data['districts']['CD_MUN'] == mun_code
                    ]['NM_DIST'].dropna().sort_values().unique()
                    
                    if len(available_districts) > 0:
                        selected_district = st.selectbox(
                            "Choose a district:",
                            options=[None] + list(available_districts),
                            format_func=lambda x: "Select district..." if x is None else x,
                            key="district_selector"
                        )
                        
                        # Reset neighborhood when district changes
                        if selected_district != st.session_state.selected_district:
                            st.session_state.selected_district = selected_district
                            st.session_state.selected_neighborhood = None
                    else:
                        st.info("No districts available")
                        selected_district = None
                else:
                    st.info("City not found")
                    selected_district = None
            else:
                st.info("Select a municipality first")
                selected_district = None
        
        # Neighborhood Dropdown (only show if district selected)
        with col4:
            st.subheader("🏠 Neighborhood")
            if selected_district:
                # Find neighborhoods for this district
                # First check what columns are available in neighborhood data
                neighborhood_df = hierarchy_data['neighborhoods']
                
                if 'NM_BAIRRO' in neighborhood_df.columns:
                    # Use direct neighborhood data filtering
                    if all(col in neighborhood_df.columns for col in ['NM_UF', 'NM_MUN', 'NM_DIST']):
                        # Filter using hierarchy columns
                        available_neighborhoods = neighborhood_df[
                            (neighborhood_df['NM_UF'] == selected_state) &
                            (neighborhood_df['NM_MUN'] == selected_city) &
                            (neighborhood_df['NM_DIST'] == selected_district)
                        ]['NM_BAIRRO'].dropna().sort_values().unique()
                    else:
                        # Fallback: use district code to find neighborhoods
                        # Get the district code first
                        city_info = municipality_data[
                            (municipality_data['state'] == selected_state) & 
                            (municipality_data['name'] == selected_city)
                        ]
                        
                        if len(city_info) > 0:
                            mun_code = city_info['CD_MUN'].iloc[0]
                            district_info = hierarchy_data['districts'][
                                (hierarchy_data['districts']['CD_MUN'] == mun_code) &
                                (hierarchy_data['districts']['NM_DIST'] == selected_district)
                            ]
                            
                            if len(district_info) > 0:
                                district_code = district_info['CD_DIST'].iloc[0]
                                # Find neighborhoods by district code pattern
                                if 'CD_BAIRRO' in neighborhood_df.columns:
                                    available_neighborhoods = neighborhood_df[
                                        neighborhood_df['CD_BAIRRO'].str.startswith(district_code[:9])  # First 9 chars of district code
                                    ]['NM_BAIRRO'].dropna().sort_values().unique()
                                else:
                                    available_neighborhoods = []
                            else:
                                available_neighborhoods = []
                        else:
                            available_neighborhoods = []
                else:
                    available_neighborhoods = []
                
                if len(available_neighborhoods) > 0:
                    selected_neighborhood = st.selectbox(
                        "Choose a neighborhood:",
                        options=[None] + list(available_neighborhoods),
                        format_func=lambda x: "Select neighborhood..." if x is None else x,
                        key="neighborhood_selector"
                    )
                    st.session_state.selected_neighborhood = selected_neighborhood
                else:
                    st.info("No neighborhoods available")
                    selected_neighborhood = None
            else:
                st.info("Select a district first")  
                selected_neighborhood = None
        
    with tab2:
        # Side-by-side Comparison Feature
        st.header("⚖️ Side-by-Side Comparison")
        
        # Initialize comparison session state
        if 'comparison_items' not in st.session_state:
            st.session_state.comparison_items = []
        if 'comparison_level' not in st.session_state:
            st.session_state.comparison_level = 'municipality'
        
        # Comparison level selector
        col1, col2 = st.columns([1, 2])
        
        with col1:
            comparison_level = st.selectbox(
                "Compare:",
                options=['municipality', 'district', 'neighborhood'],
                format_func=lambda x: x.title() + 's',
                key="comparison_level_selector"
            )
            st.session_state.comparison_level = comparison_level
        
        with col2:
            # Add item to comparison
            if comparison_level == 'municipality':
                # Municipality comparison
                available_states = sorted(municipality_data['state'].dropna().unique())
                comp_state = st.selectbox(
                    "Select state:",
                    options=available_states,
                    key="comp_state"
                )
                
                if comp_state:
                    available_cities = municipality_data[
                        municipality_data['state'] == comp_state
                    ]['name'].dropna().sort_values().unique()
                    
                    comp_city = st.selectbox(
                        "Select municipality:",
                        options=[None] + list(available_cities),
                        format_func=lambda x: "Select municipality..." if x is None else x,
                        key="comp_city"
                    )
                    
                    if comp_city and st.button("➕ Add to Comparison"):
                        # Get municipality data
                        city_data = municipality_data[
                            (municipality_data['state'] == comp_state) &
                            (municipality_data['name'] == comp_city)
                        ].iloc[0] if len(municipality_data[
                            (municipality_data['state'] == comp_state) &
                            (municipality_data['name'] == comp_city)
                        ]) > 0 else None
                        
                        if city_data is not None:
                            city_data_dict = city_data.to_dict()
                            city_data_dict['display_name'] = f"{comp_city}, {comp_state}"
                            city_data_dict['level'] = 'municipality'
                            city_data_dict['elders_above_8k'] = calculate_elders_above_8k(
                                city_data_dict['elderly_total'], 
                                city_data_dict['avg_income'], 
                                city_data_dict['income_std']
                            )
                            
                            # Check if already in comparison
                            existing = any(item['display_name'] == city_data_dict['display_name'] 
                                         for item in st.session_state.comparison_items)
                            
                            if not existing:
                                st.session_state.comparison_items.append(city_data_dict)
                                st.success(f"Added {city_data_dict['display_name']} to comparison")
                            else:
                                st.warning(f"{city_data_dict['display_name']} already in comparison")
            
            elif comparison_level == 'district':
                # District comparison
                comp_state_dist = st.selectbox(
                    "Select state:",
                    options=available_states,
                    key="comp_state_dist"
                )
                
                if comp_state_dist:
                    available_cities_dist = municipality_data[
                        municipality_data['state'] == comp_state_dist
                    ]['name'].dropna().sort_values().unique()
                    
                    comp_city_dist = st.selectbox(
                        "Select municipality:",
                        options=[None] + list(available_cities_dist),
                        format_func=lambda x: "Select municipality..." if x is None else x,
                        key="comp_city_dist"
                    )
                    
                    if comp_city_dist:
                        # Get municipality code
                        city_info_dist = municipality_data[
                            (municipality_data['state'] == comp_state_dist) & 
                            (municipality_data['name'] == comp_city_dist)
                        ]
                        
                        if len(city_info_dist) > 0:
                            mun_code_dist = city_info_dist['CD_MUN'].iloc[0]
                            available_districts_comp = hierarchy_data['districts'][
                                hierarchy_data['districts']['CD_MUN'] == mun_code_dist
                            ]['NM_DIST'].dropna().sort_values().unique()
                            
                            comp_district = st.selectbox(
                                "Select district:",
                                options=[None] + list(available_districts_comp),
                                format_func=lambda x: "Select district..." if x is None else x,
                                key="comp_district"
                            )
                            
                            if comp_district and st.button("➕ Add District to Comparison"):
                                # Get district data
                                district_data = hierarchy_data['districts'][
                                    (hierarchy_data['districts']['CD_MUN'] == mun_code_dist) &
                                    (hierarchy_data['districts']['NM_DIST'] == comp_district)
                                ]
                                
                                if len(district_data) > 0:
                                    district_dict = district_data.iloc[0].to_dict()
                                    district_dict['display_name'] = f"{comp_district}, {comp_city_dist}, {comp_state_dist}"
                                    district_dict['level'] = 'district'
                                    district_dict['elders_above_8k'] = calculate_elders_above_8k(
                                        district_dict['elderly_total'], 
                                        district_dict['avg_income'], 
                                        district_dict['income_std']
                                    )
                                    
                                    # Check if already in comparison
                                    existing = any(item['display_name'] == district_dict['display_name'] 
                                                 for item in st.session_state.comparison_items)
                                    
                                    if not existing:
                                        st.session_state.comparison_items.append(district_dict)
                                        st.success(f"Added {district_dict['display_name']} to comparison")
                                    else:
                                        st.warning(f"{district_dict['display_name']} already in comparison")
            
            elif comparison_level == 'neighborhood':
                # Neighborhood comparison
                comp_state_nbr = st.selectbox(
                    "Select state:",
                    options=available_states,
                    key="comp_state_nbr"
                )
                
                if comp_state_nbr:
                    available_cities_nbr = municipality_data[
                        municipality_data['state'] == comp_state_nbr
                    ]['name'].dropna().sort_values().unique()
                    
                    comp_city_nbr = st.selectbox(
                        "Select municipality:",
                        options=[None] + list(available_cities_nbr),
                        format_func=lambda x: "Select municipality..." if x is None else x,
                        key="comp_city_nbr"
                    )
                    
                    if comp_city_nbr:
                        # Get districts for this city
                        city_info_nbr = municipality_data[
                            (municipality_data['state'] == comp_state_nbr) & 
                            (municipality_data['name'] == comp_city_nbr)
                        ]
                        
                        if len(city_info_nbr) > 0:
                            mun_code_nbr = city_info_nbr['CD_MUN'].iloc[0]
                            available_districts_nbr = hierarchy_data['districts'][
                                hierarchy_data['districts']['CD_MUN'] == mun_code_nbr
                            ]['NM_DIST'].dropna().sort_values().unique()
                            
                            if len(available_districts_nbr) > 0:
                                comp_district_nbr = st.selectbox(
                                    "Select district:",
                                    options=[None] + list(available_districts_nbr),
                                    format_func=lambda x: "Select district..." if x is None else x,
                                    key="comp_district_nbr"
                                )
                                
                                if comp_district_nbr:
                                    # Get neighborhoods for this district
                                    neighborhood_df = hierarchy_data['neighborhoods']
                                    if all(col in neighborhood_df.columns for col in ['NM_UF', 'NM_MUN', 'NM_DIST']):
                                        available_nbr = neighborhood_df[
                                            (neighborhood_df['NM_UF'] == comp_state_nbr) &
                                            (neighborhood_df['NM_MUN'] == comp_city_nbr) &
                                            (neighborhood_df['NM_DIST'] == comp_district_nbr)
                                        ]['NM_BAIRRO'].dropna().sort_values().unique()
                                    else:
                                        available_nbr = []
                                    
                                    if len(available_nbr) > 0:
                                        comp_neighborhood = st.selectbox(
                                            "Select neighborhood:",
                                            options=[None] + list(available_nbr),
                                            format_func=lambda x: "Select neighborhood..." if x is None else x,
                                            key="comp_neighborhood"
                                        )
                                        
                                        if comp_neighborhood and st.button("➕ Add Neighborhood to Comparison"):
                                            # Get neighborhood data
                                            if 'NM_BAIRRO' in neighborhood_df.columns:
                                                nbr_name_col = 'NM_BAIRRO'
                                            elif 'NM_BAIRRO_x' in neighborhood_df.columns:
                                                nbr_name_col = 'NM_BAIRRO_x'
                                            else:
                                                nbr_name_col = 'NM_BAIRRO'
                                                
                                            nbr_data = neighborhood_df[
                                                (neighborhood_df['NM_UF'] == comp_state_nbr) &
                                                (neighborhood_df['NM_MUN'] == comp_city_nbr) &
                                                (neighborhood_df['NM_DIST'] == comp_district_nbr) &
                                                (neighborhood_df[nbr_name_col] == comp_neighborhood)
                                            ]
                                            
                                            if len(nbr_data) > 0:
                                                nbr_dict = nbr_data.iloc[0].to_dict()
                                                nbr_dict['display_name'] = f"{comp_neighborhood}, {comp_district_nbr}, {comp_city_nbr}, {comp_state_nbr}"
                                                nbr_dict['level'] = 'neighborhood'
                                                nbr_dict['elders_above_8k'] = calculate_elders_above_8k(
                                                    nbr_dict['elderly_total'], 
                                                    nbr_dict['avg_income'], 
                                                    nbr_dict['income_std']
                                                )
                                                
                                                # Check if already in comparison
                                                existing = any(item['display_name'] == nbr_dict['display_name'] 
                                                             for item in st.session_state.comparison_items)
                                                
                                                if not existing:
                                                    st.session_state.comparison_items.append(nbr_dict)
                                                    st.success(f"Added {nbr_dict['display_name']} to comparison")
                                                else:
                                                    st.warning(f"{nbr_dict['display_name']} already in comparison")
                                    else:
                                        st.info("No neighborhoods available for comparison")
                            else:
                                st.info("No districts available for this city")
        
        # Display current comparison items
        if st.session_state.comparison_items:
            st.subheader(f"📊 Comparing {len(st.session_state.comparison_items)} {comparison_level}s")
            
            # Clear comparison button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️ Clear Comparison"):
                    st.session_state.comparison_items = []
                    st.rerun()
            
            # Create comparison table
            if len(st.session_state.comparison_items) > 0:
                # Comparison table
                st.subheader("📋 Detailed Comparison Table")
                
                comparison_df = pd.DataFrame(st.session_state.comparison_items)
                
                # Select columns to display
                display_columns = ['display_name', 'elderly_total', 'avg_income', 'income_std', 'elders_above_8k']
                if 'total_population' in comparison_df.columns:
                    display_columns.insert(2, 'total_population')
                    comparison_df['elderly_percentage'] = np.where(
                        comparison_df['total_population'] > 0,
                        (comparison_df['elderly_total'] / comparison_df['total_population'] * 100).round(1),
                        0
                    )
                    display_columns.insert(4, 'elderly_percentage')
                
                # Filter available columns
                available_display_cols = [col for col in display_columns if col in comparison_df.columns]
                comparison_display = comparison_df[available_display_cols].copy()
                
                # Rename columns
                column_names = {
                    'display_name': 'Location',
                    'total_population': 'Total Population', 
                    'elderly_total': 'Elderly Pop (70+)',
                    'elderly_percentage': 'Elderly %',
                    'avg_income': 'Avg Income (R$)',
                    'income_std': 'Income Std Dev (R$)',
                    'elders_above_8k': 'Elders Above R$ 8k'
                }
                
                comparison_display = comparison_display.rename(columns=column_names)

                # Configure column formatting (keeps numeric values for proper sorting)
                comparison_column_config = {}
                for col in comparison_display.columns:
                    if 'Population' in col and col != 'Elderly %':
                        comparison_column_config[col] = st.column_config.NumberColumn(
                            col,
                            format="%d",
                            help=None
                        )
                    elif 'Elderly %' in col:
                        comparison_column_config[col] = st.column_config.NumberColumn(
                            col,
                            format="%.1f%%",
                            help=None
                        )
                    elif 'Income' in col or 'R$' in col:
                        comparison_column_config[col] = st.column_config.NumberColumn(
                            col,
                            format="R$ %.2f",
                            help=None
                        )
                    elif 'Elders Above R$ 8k' in col:
                        comparison_column_config[col] = st.column_config.NumberColumn(
                            col,
                            format="%d",
                            help=None
                        )

                st.dataframe(
                    comparison_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config=comparison_column_config
                )
                
                # Export comparison
                if st.button("📥 Export Comparison to CSV"):
                    csv_data = pd.DataFrame(st.session_state.comparison_items)[available_display_cols]
                    csv_data = csv_data.rename(columns=column_names)
                    csv = csv_data.to_csv(index=False)
                    st.download_button(
                        label="Download Comparison CSV",
                        data=csv,
                        file_name=f"{comparison_level}_comparison.csv",
                        mime="text/csv"
                    )
    
    # Move main analysis back to tab1
    with tab1:
        st.markdown("---")
        
        # Determine what level to display and filter data accordingly  
        if selected_neighborhood:
                # Show specific neighborhood data
                current_level = "Neighborhood"
                filtered_data = hierarchy_data['neighborhoods'][
                    (hierarchy_data['neighborhoods']['NM_UF'] == selected_state) &
                    (hierarchy_data['neighborhoods']['NM_MUN'] == selected_city) &
                    (hierarchy_data['neighborhoods']['NM_DIST'] == selected_district) &
                    (hierarchy_data['neighborhoods']['NM_BAIRRO'] == selected_neighborhood)
                ].copy()
                
                # Add elders above 8k calculation
                if len(filtered_data) > 0:
                    filtered_data['elders_above_8k'] = filtered_data.apply(
                        lambda row: calculate_elders_above_8k(
                            row['elderly_total'], row['avg_income'], row['income_std']
                        ), axis=1
                    )
                
                display_name = f"{selected_neighborhood}, {selected_district}, {selected_city}, {selected_state}"
                
        elif selected_district:
                # Show all neighborhoods in the district
                current_level = "Neighborhoods in District"
                filtered_data = hierarchy_data['neighborhoods'][
                    (hierarchy_data['neighborhoods']['NM_UF'] == selected_state) &
                    (hierarchy_data['neighborhoods']['NM_MUN'] == selected_city) &
                    (hierarchy_data['neighborhoods']['NM_DIST'] == selected_district)
                ].copy()
                
                # Add elders above 8k calculation
                filtered_data['elders_above_8k'] = filtered_data.apply(
                    lambda row: calculate_elders_above_8k(
                        row['elderly_total'], row['avg_income'], row['income_std']
                    ), axis=1
                )
                
                display_name = f"All neighborhoods in {selected_district}, {selected_city}, {selected_state}"
                
        elif selected_city:
                # Show all districts in the city
                current_level = "Districts in Municipality"
                city_info = municipality_data[
                    (municipality_data['state'] == selected_state) & 
                    (municipality_data['name'] == selected_city)
                ]
                
                if len(city_info) > 0:
                    mun_code = city_info['CD_MUN'].iloc[0]
                    filtered_data = hierarchy_data['districts'][
                        hierarchy_data['districts']['CD_MUN'] == mun_code
                    ].copy()
                    
                    # Add elders above 8k calculation
                    filtered_data['elders_above_8k'] = filtered_data.apply(
                        lambda row: calculate_elders_above_8k(
                            row['elderly_total'], row['avg_income'], row['income_std']
                        ), axis=1
                    )
                else:
                    filtered_data = pd.DataFrame()
                    
                display_name = f"All districts in {selected_city}, {selected_state}"
                
        elif selected_state:
                # Show all municipalities in the state
                current_level = "Municipalities in State"
                filtered_data = municipality_data[
                    municipality_data['state'] == selected_state
                ].copy()
                
                # Add elders above 8k calculation
                filtered_data['elders_above_8k'] = filtered_data.apply(
                    lambda row: calculate_elders_above_8k(
                        row['elderly_total'], row['avg_income'], row['income_std']
                    ), axis=1
                )
                
                display_name = f"All municipalities in {selected_state}"
                
        else:
                # Show all municipalities (default view)
                current_level = "All Municipalities"
                filtered_data = municipality_data.copy()
                
                # Add elders above 8k calculation  
                filtered_data['elders_above_8k'] = filtered_data.apply(
                    lambda row: calculate_elders_above_8k(
                        row['elderly_total'], row['avg_income'], row['income_std']
                    ), axis=1
                )
                
                display_name = "All Brazilian Municipalities"
            
        # Display results
        st.header(f"📊 {current_level}: {display_name}")
        st.write(f"**{len(filtered_data)}** results found")
        
        if len(filtered_data) == 0:
            st.warning("No data available for the selected geographic area.")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_elderly = filtered_data['elderly_total'].sum()
            st.metric("Total Elderly Population (70+)", f"{total_elderly:,.0f}")
        with col2:
            avg_income = np.average(
                filtered_data['avg_income'].dropna(), 
                weights=filtered_data.loc[filtered_data['avg_income'].notna(), 'elderly_total'].fillna(1)
            ) if len(filtered_data['avg_income'].dropna()) > 0 else 0
            st.metric("Weighted Avg Income", f"R$ {avg_income:,.2f}")
        with col3:
            total_elders_8k = filtered_data['elders_above_8k'].sum()
            st.metric("Total Elders Above R$ 8k", f"{total_elders_8k:,.0f}")
        with col4:
            level_name = current_level.split()[0] if current_level else "Areas"
            st.metric(f"Number of {level_name}", f"{len(filtered_data):,}")
        
        # Determine display columns based on geographic level
        if current_level == "Neighborhood":
            # Single neighborhood view
            display_cols = ['elderly_total', 'avg_income', 'income_std', 'elders_above_8k']
            col_names = ['Elderly Population (70+)', 'Average Income (R$)', 'Income Std Dev (R$)', 'Elders Above R$ 8k']
        elif current_level == "Neighborhoods in District":
            # Multiple neighborhoods
            # Check what name column is available
            if 'NM_BAIRRO' in filtered_data.columns:
                name_col = 'NM_BAIRRO'
            elif 'NM_BAIRRO_x' in filtered_data.columns:
                name_col = 'NM_BAIRRO_x'
            elif 'NM_BAIRRO_y' in filtered_data.columns:
                name_col = 'NM_BAIRRO_y'
            else:
                name_col = filtered_data.columns[0]  # Use first column as fallback
                
            display_cols = [name_col, 'elderly_total', 'avg_income', 'income_std', 'elders_above_8k']  
            col_names = ['Neighborhood', 'Elderly Pop (70+)', 'Avg Income (R$)', 'Income Std Dev (R$)', 'Elders Above R$ 8k']
        elif current_level == "Districts in Municipality":
            # Multiple districts
            name_col = 'NM_DIST'
            display_cols = [name_col, 'elderly_total', 'avg_income', 'income_std', 'elders_above_8k']
            col_names = ['District', 'Elderly Pop (70+)', 'Avg Income (R$)', 'Income Std Dev (R$)', 'Elders Above R$ 8k']
        else:
            # Municipalities (default)
            name_col = 'name'
            if 'total_population' in filtered_data.columns and 'elderly_percentage' in filtered_data.columns:
                display_cols = [name_col, 'state', 'total_population', 'elderly_total', 'elderly_percentage', 'avg_income', 'income_std', 'elders_above_8k']
                col_names = ['Municipality', 'State', 'Total Population', 'Elderly Pop (70+)', 'Elderly %', 'Avg Income (R$)', 'Income Std Dev (R$)', 'Elders Above R$ 8k']
            else:
                display_cols = [name_col, 'state', 'elderly_total', 'avg_income', 'income_std', 'elders_above_8k']
                col_names = ['Municipality', 'State', 'Elderly Pop (70+)', 'Avg Income (R$)', 'Income Std Dev (R$)', 'Elders Above R$ 8k']
        
        # Filter only available columns
        available_cols = [col for col in display_cols if col in filtered_data.columns]
        available_col_names = [col_names[i] for i, col in enumerate(display_cols) if col in filtered_data.columns]
        
        display_data = filtered_data[available_cols].copy()
        
        # Ensure numeric columns are properly typed before sorting (only for existing columns)
        numeric_cols = ['elderly_total', 'avg_income', 'income_std', 'elders_above_8k']
        if 'total_population' in display_data.columns:
            numeric_cols.append('total_population')
        if 'elderly_percentage' in display_data.columns:
            numeric_cols.append('elderly_percentage')
            
        for col in numeric_cols:
            if col in display_data.columns:
                display_data[col] = pd.to_numeric(display_data[col], errors='coerce').fillna(0)
        
        # Sort by market score (numeric sort)
        display_data = display_data.sort_values('elders_above_8k', ascending=False)
        
        # Rename columns for display
        display_data.columns = available_col_names

        # Configure column formatting (keeps numeric values for proper sorting)
        column_config = {}
        for col in display_data.columns:
            if 'Population' in col and col != 'Elderly %':
                column_config[col] = st.column_config.NumberColumn(
                    col,
                    format="%d",
                    help=None
                )
            elif 'Elderly %' in col:
                column_config[col] = st.column_config.NumberColumn(
                    col,
                    format="%.1f%%",
                    help=None
                )
            elif 'Income' in col or 'R$' in col:
                column_config[col] = st.column_config.NumberColumn(
                    col,
                    format="R$ %.2f",
                    help=None
                )
            elif 'Elders Above R$ 8k' in col:
                column_config[col] = st.column_config.NumberColumn(
                    col,
                    format="%d",
                    help=None
                )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )
        
        # Export functionality
        if st.button("📥 Export Results to CSV"):
            csv = filtered_data[available_cols].to_csv(index=False)
            file_name = f"{current_level.lower().replace(' ', '_')}_analysis.csv"
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=file_name,
                mime="text/csv"
            )
        
        # Additional filters for large datasets
        if len(filtered_data) > 50:
            st.markdown("---")
            st.subheader("🔍 Additional Filters")

            col1, col2 = st.columns(2)
            with col1:
                min_elderly = st.number_input("Minimum Elderly Population (70+):", min_value=0, value=0, key="filter_elderly")
            with col2:
                min_income = st.number_input("Minimum Average Income (BRL):", min_value=0.0, value=0.0, step=100.0, key="filter_income")

            if min_elderly > 0 or min_income > 0:
                # Filter using numeric values directly
                elderly_col = 'Elderly Pop (70+)'
                income_col = 'Avg Income (R$)'

                mask = pd.Series([True] * len(display_data))
                if min_elderly > 0 and elderly_col in display_data.columns:
                    mask = mask & (display_data[elderly_col] >= min_elderly)
                if min_income > 0 and income_col in display_data.columns:
                    mask = mask & (display_data[income_col] >= min_income)

                additional_filtered = display_data[mask]

                st.write(f"**{len(additional_filtered)}** results match additional filters")
                if len(additional_filtered) > 0:
                    st.dataframe(
                        additional_filtered,
                        use_container_width=True,
                        hide_index=True,
                        column_config=column_config
                    )

if __name__ == "__main__":
    main()