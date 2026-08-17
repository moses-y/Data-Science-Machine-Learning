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
    print("Columns in biochar_merged after merging with agent_lookup:", biochar_merged.columns)

    # Check for any unmatched agent_ids
    unmatched_agents = biochar_merged[biochar_merged['_merge'] != 'both']
    if not unmatched_agents.empty:
        print("Warning: Some agent_ids in Biochar Sales do not match those in Agent Lookup.")
        print(unmatched_agents[['agent_id']].drop_duplicates())
    biochar_merged.drop(columns=['_merge'], inplace=True)

    # Merge with County Lookup
    biochar_merged = pd.merge(
        biochar_merged,
        county_lookup,
        left_on='county_id_y',
        right_on='county_id',
        how='left',
        indicator=True
    )

    # Check for any unmatched county_ids
    unmatched_counties = biochar_merged[biochar_merged['_merge'] != 'both']
    if not unmatched_counties.empty:
        print("Warning: Some county_ids in Biochar Sales do not match those in County Lookup.")
        print(unmatched_counties[['county_id_y']].drop_duplicates())
    biochar_merged.drop(columns=['_merge'], inplace=True)

    # Calculate total revenue expected
    biochar_merged['revenue_expected'] = biochar_merged['biochar_bags'] * 1800

    # Calculate remaining balance
    biochar_merged['remaining_balance'] = biochar_merged['revenue_expected'] - biochar_merged['paid_today']

    # Group by county and agent, then aggregate
    biochar_summary = biochar_merged.groupby(['county_name', 'agent_name']).agg(
        total_revenue_expected=('revenue_expected', 'sum'),
        total_revenue_received=('paid_today', 'sum'),
        total_bags_sold=('biochar_bags', 'sum'),
        total_remaining_balance=('remaining_balance', 'sum')
    ).reset_index()

    # Sort within each county by total revenue expected (descending)
    biochar_summary_sorted = biochar_summary.sort_values(
        by=['county_name', 'total_revenue_expected'],
        ascending=[True, False]
    )

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

    # Check for any unmatched agent_ids
    unmatched_agents_para = parasitoid_merged[parasitoid_merged['_merge'] != 'both']
    if not unmatched_agents_para.empty:
        print("Warning: Some agent_ids in Parasitoid Sales Orders do not match those in Agent Lookup.")
        print(unmatched_agents_para[['agent_id']].drop_duplicates())
    parasitoid_merged.drop(columns=['_merge'], inplace=True)

    # Merge with County Lookup
    parasitoid_merged = pd.merge(
        parasitoid_merged,
        county_lookup,
        left_on='county_id_x',
        right_on='county_id',
        how='left',
        indicator=True
    )

    # Check for any unmatched county_ids
    unmatched_counties_para = parasitoid_merged[parasitoid_merged['_merge'] != 'both']
    if not unmatched_counties_para.empty:
        print("Warning: Some county_ids in Parasitoid Sales Orders do not match those in County Lookup.")
        print(unmatched_counties_para[['county_id_x']].drop_duplicates())
    parasitoid_merged.drop(columns=['_merge'], inplace=True)

    # Calculate total revenue expected
    parasitoid_merged['revenue_expected'] = parasitoid_merged['order_amount']

    # Merge with Parasitoids Payment
    parasitoid_merged = pd.merge(
        parasitoid_merged,
        parasitoids_payment[['farmer_id', 'paid_today']],
        on='farmer_id',
        how='left',
        indicator=True
    )

    # Check for any unmatched farmer_ids
    unmatched_farmers = parasitoid_merged[parasitoid_merged['_merge'] != 'both']
    if not unmatched_farmers.empty:
        print("Warning: Some farmer_ids in Parasitoid Sales Orders do not match those in Parasitoids Payment.")
        print(unmatched_farmers[['farmer_id']].drop_duplicates())
    parasitoid_merged.drop(columns=['_merge'], inplace=True)

    # Replace NaN in paid_today with 0
    parasitoid_merged['paid_today'] = parasitoid_merged['paid_today'].fillna(0)

    # Calculate remaining balance
    parasitoid_merged['remaining_balance'] = parasitoid_merged['revenue_expected'] - parasitoid_merged['paid_today']

    # Calculate total cards sold
    parasitoid_merged['total_cards_sold'] = parasitoid_merged['order_amount'] / 299

    # Group by county and agent, then aggregate
    parasitoid_summary = parasitoid_merged.groupby(['county_name', 'agent_name']).agg(
        total_revenue_expected=('revenue_expected', 'sum'),
        total_revenue_received=('paid_today', 'sum'),
        total_cards_sold=('total_cards_sold', 'sum'),
        total_remaining_balance=('remaining_balance', 'sum')
    ).reset_index()

    # Sort within each county by total revenue expected (descending)
    parasitoid_summary_sorted = parasitoid_summary.sort_values(
        by=['county_name', 'total_revenue_expected'],
        ascending=[True, False]
    )

except KeyError as e:
    print(f"Error in Parasitoid Sales processing: Missing column {e}")
    exit(1)

# --- Function to Format Pivot Table Data ---
def format_pivot_table(df, product_type, notes):
    formatted_data = [([f"Part: {product_type} Sales Summary"])] # Title for the table
    # Add headers for clarity in Excel
    headers = ['Name', 'Revenue Expected', 'Revenue Received', 'Remaining Balance',
               f'Total {product_type} {"Bags" if product_type == "Biochar" else "Cards"}']
    formatted_data.append(headers)

    # Process each county
    for county in df['county_name'].unique():
        county_data = df[df['county_name'] == county]

        # Add county name as a separate row
        formatted_data.append([county, '', '', '', ''])

        # Add agent data
        for _, row in county_data.iterrows():
            formatted_data.append([
                row['agent_name'],
                format_currency(row['total_revenue_expected']),
                format_currency(row['total_revenue_received']),
                format_currency(row['total_remaining_balance']),
                int(row['total_bags_sold']) if product_type == 'Biochar' else round(row['total_cards_sold'], 0)
            ])

        # Calculate and add county subtotal
        subtotal = county_data.agg({
            'total_revenue_expected': 'sum',
            'total_revenue_received': 'sum',
            'total_remaining_balance': 'sum',
            'total_bags_sold' if product_type == 'Biochar' else 'total_cards_sold': 'sum'
        })

        formatted_data.append([
            f"{county} Sub_total",
            format_currency(subtotal['total_revenue_expected']),
            format_currency(subtotal['total_revenue_received']),
            format_currency(subtotal['total_remaining_balance']),
            int(subtotal['total_bags_sold']) if product_type == 'Biochar' else round(subtotal['total_cards_sold'], 0)
        ])

        # Add a blank row for separation
        formatted_data.append(['', '', '', '', ''])

    # Calculate and add grand total
    grand_total = df.agg({
        'total_revenue_expected': 'sum',
        'total_revenue_received': 'sum',
        'total_remaining_balance': 'sum',
        'total_bags_sold' if product_type == 'Biochar' else 'total_cards_sold': 'sum'
    })

    formatted_data.append([
        'Grand Total',
        format_currency(grand_total['total_revenue_expected']),
        format_currency(grand_total['total_revenue_received']),
        format_currency(grand_total['total_remaining_balance']),
        int(grand_total['total_bags_sold']) if product_type == 'Biochar' else round(grand_total['total_cards_sold'], 0)
    ])

    # Add notes
    formatted_data.append(['']) # Blank line before notes
    formatted_data.append(['Observations:'])
    for note in notes:
        formatted_data.append([note])

    formatted_data.append(['']) # Extra blank line after notes
    return formatted_data

# --- Creating the Pivot Tables and Writing to Excel ---
try:
    # Define notes for each pivot table
    biochar_notes = [
        f"- Biochar sales show a total expected revenue of Kshs. {biochar_summary_sorted['total_revenue_expected'].sum():,.2f}.",
        "- Further analysis can be done to understand the remaining balances and agent performance."
    ]
    parasitoid_notes = [
        f"- Parasitoid sales indicate a total expected revenue of Kshs. {parasitoid_summary_sorted['total_revenue_expected'].sum():,.2f}.",
        "- Consider strategies to improve payment collection for parasitoid orders."
    ]

    # Format pivot table data with notes
    biochar_data = format_pivot_table(biochar_summary_sorted, 'Biochar', biochar_notes)
    parasitoid_data = format_pivot_table(parasitoid_summary_sorted, 'Parasitoid', parasitoid_notes)

    # Create Excel writer using openpyxl as the engine
    with pd.ExcelWriter('your_name_excel_response.xlsx', engine='openpyxl') as writer:
        startrow = 0
        # Write Biochar pivot table
        biochar_df_output = pd.DataFrame(biochar_data)
        biochar_df_output.to_excel(
            writer,
            sheet_name='pivot_table',
            index=False,
            header=False,
            startrow=startrow
        )
        print("Biochar Fertilizer Sales Summary pivot table written successfully to 'pivot_table' sheet.")
        startrow += len(biochar_data) + 2 # Add space between tables

        # Write Parasitoid pivot table
        parasitoid_df_output = pd.DataFrame(parasitoid_data)
        parasitoid_df_output.to_excel(
            writer,
            sheet_name='pivot_table',
            index=False,
            header=False,
            startrow=startrow
        )
        print("Parasitoid Sales Summary pivot table written successfully to 'pivot_table' sheet.")

except Exception as e:
    print(f"An error occurred while writing to Excel: {e}")
    exit(1)

print("\nExcel file 'your_name_excel_response.xlsx' has been created successfully.")
# Created/Modified files during execution:
print('your_name_excel_response.xlsx')