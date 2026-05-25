import pandas as pd

from db import engine


try:

    # Read Excel
    df = pd.read_excel(
        "employees_dataset.xlsx"
    )

    # Upload to Postgres
    df.to_sql(
        name="employees",
        con=engine,
        if_exists="replace",
        index=False
    )

    print(
        "Employees dataset uploaded successfully!"
    )

    print(
        f"Rows uploaded: {len(df)}"
    )

except Exception as e:

    print(
        "Error:",
        str(e)
    )