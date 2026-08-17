import pandas as pd
import openpyxl

# Function to format currency
def format_currency(value):
    return f"Kshs. {value:,.2f}"

# Load the Excel file and list available sheets
try:
    excel_file = pd.ExcelFile("Excel Assignment.xlsx")
    print("Available Sheets:", excel_file.sheet_names)
except FileNotFoundError:
    print("Error: 'Excel Assignment.xlsx' file not found in the current directory.")
    exit(1)

# Verify and parse each required sheet
required_sheets = [
    "county_lookup",
    "agent_lookup",
    "biochar_sales",
    "parasitoids_sales_orders",
    "parasitoids_sales_payment"
]

parsed_sheets = {}
for sheet in required_sheets:
    if sheet in excel_file.sheet_names:
        parsed_sheets[sheet] = excel_file.parse(sheet)
        print(f"Successfully loaded sheet: '{sheet}'")
        print(f"Columns in {sheet} DataFrame:", parsed_sheets[sheet].columns)
    else:
        print(f"Error: Worksheet named '{sheet}' not found.")
        exit(1)

# Assigning DataFrames for easier reference
county_lookup = parsed_sheets["county_lookup"]
agent_lookup = parsed_sheets["agent_lookup"]
biochar_sales = parsed_sheets["biochar_sales"]
parasitoid_sales_orders = parsed_sheets["parasitoids_sales_orders"]
parasitoids_payment = parsed_sheets["parasitoids_sales_payment"]

# --- Part 1: Biochar Fertilizer Sales Summary ---
try:
    # Merge Biochar Sales with Agent Lookup
    biochar_merged = pd.merge(
        biochar_sales,
        agent_lookup,
        on='agent_id',
        how='left',
        indicator=True
    )
    # Merge with County Lookup
    biochar_merged = pd.merge(
        biochar_merged,
        county_lookup,
        left_on='county_id_y',
        right_on='county_id',
        how='left'
    )
    biochar_merged['revenue_expected'] = biochar_merged['biochar_bags'] * 1800
    biochar_merged['remaining_balance'] = biochar_merged['revenue_expected'] - biochar_merged['paid_today']
    biochar_summary_agent = biochar_merged.groupby('agent_name').agg(
        total_revenue_expected=('revenue_expected', 'sum'),
        total_revenue_received=('paid_today', 'sum'),
        total_bags_sold=('biochar_bags', 'sum'),
        total_remaining_balance=('remaining_balance', 'sum')
    ).reset_index()
except KeyError as e:
    print(f"Error in Biochar Sales processing: Missing column {e}")
    exit(1)

# --- Part 2: Parasitoid Sales Summary ---
try:
    # Merge Parasitoid Sales Orders with Agent Lookup
    parasitoid_merged = pd.merge(
        parasitoid_sales_orders,
        agent_lookup,
        on='agent_id',
        how='left',
        indicator=True
    )
    # Remove the _merge column if you don't need it
    if '_merge' in parasitoid_merged.columns:
        parasitoid_merged.drop(columns=['_merge'], inplace=True)

    # Merge with County Lookup
    parasitoid_merged = pd.merge(
        parasitoid_merged,
        county_lookup,
        left_on='county_id_x',
        right_on='county_id',
        how='left'
    )
    parasitoid_merged['revenue_expected'] = parasitoid_merged['order_amount']

    # Merge with Parasitoids Payment
    parasitoid_merged = pd.merge(
        parasitoid_merged,
        parasitoids_payment[['farmer_id', 'paid_today']],
        on='farmer_id',
        how='left'
    )

    # Replace NaN in paid_today with 0
    parasitoid_merged['paid_today'] = parasitoid_merged['paid_today'].fillna(0)

    # Calculate remaining balance
    parasitoid_merged['remaining_balance'] = parasitoid_merged['revenue_expected'] - parasitoid_merged['paid_today']

    # Calculate total_cards_sold
    parasitoid_merged['total_cards_sold'] = parasitoid_merged['order_amount'] / 299

    parasitoid_summary_agent = parasitoid_merged.groupby('agent_name').agg(
        total_revenue_expected=('revenue_expected', 'sum'),
        total_revenue_received=('paid_today', 'sum'),
        total_cards_sold=('total_cards_sold', 'sum'),
        total_remaining_balance=('remaining_balance', 'sum')
    ).reset_index()
except KeyError as e:
    print(f"Error in Parasitoid Sales processing: Missing column {e}")
    exit(1)

# --- Create new pivot tables ---
# Agent Performance Summary
agent_performance = pd.merge(
    biochar_summary_agent,
    parasitoid_summary_agent,
    on='agent_name',
    how='outer',
    suffixes=('_biochar', '_parasitoid')
).fillna(0)

# County Sales Summary
biochar_summary_county = biochar_merged.groupby('county_name').agg(
    total_revenue_expected_biochar=('revenue_expected', 'sum'),
    total_revenue_received_biochar=('paid_today', 'sum')
).reset_index()

parasitoid_summary_county = parasitoid_merged.groupby('county_name').agg(
    total_revenue_expected_parasitoid=('revenue_expected', 'sum'),
    total_revenue_received_parasitoid=('paid_today', 'sum')
).reset_index()

county_sales = pd.merge(
    biochar_summary_county,
    parasitoid_summary_county,
    on='county_name',
    how='outer'
).fillna(0)

# --- Writing to Excel ---
output_filename = "moses_yebei_excel_response.xlsx"
try:
    with pd.ExcelWriter(output_filename, engine='openpyxl', mode='w') as writer: # Changed back to 'w'
        # Write the pivot tables to the 'pivot_table' sheet
        agent_performance.to_excel(writer, sheet_name='pivot_table', index=False, startrow=0)

        # Calculate the starting row for the County Sales table
        startrow_county_sales = len(agent_performance) + 3  # Add some space

        county_sales.to_excel(writer, sheet_name='pivot_table', index=False, startrow=startrow_county_sales)

except Exception as e:
    print(f"An error occurred while writing to Excel: {e}")
    exit(1)

print(f"\nExcel file '{output_filename}' has been created successfully.")
# Created/Modified files during execution:
print(output_filename)