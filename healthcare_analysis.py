"""
Healthcare Appointment & Revenue Analysis
-----------------------------------------
Practice project using synthetic healthcare appointment data.

Install:
    pip install pandas numpy matplotlib seaborn

Run:
    python healthcare_analysis.py

Outputs:
    healthcare_analysis.csv
    hospital_analysis.csv
    provider_analysis.csv
    healthcare.db
    appointment_status.png
    hospital_revenue.png
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_data(n=1000, seed=42):
    """Generate synthetic appointment data for practice."""
    np.random.seed(seed)

    df = pd.DataFrame({
        "Appointment_ID": range(1001, 1001 + n),
        "Patient_ID": np.random.randint(101, 401, n),
        "Hospital": np.random.choice(
            ["Apollo", "City Hospital", "Global", "Metro"], n
        ),
        "Provider": np.random.choice(
            ["Dr. Arun", "Dr. Priya", "Dr. Kumar", "Dr. Meena"], n
        ),
        "Appointment_Date": pd.date_range(
            "2026-01-01", periods=n, freq="8h"
        ),
        "Appointment_Status": np.random.choice(
            ["Completed", "Cancelled", "No Show"],
            n,
            p=[0.70, 0.15, 0.15],
        ),
        "Appointment_Type": np.random.choice(
            ["Consultation", "Follow-up", "Specialist"], n
        ),
        "Revenue": np.random.randint(300, 5000, n),
        "Payment_Status": np.random.choice(
            ["Paid", "Pending"], n, p=[0.80, 0.20]
        ),
    })

    return df


def main():
    # 1. Generate data
    df = generate_data(n=1000)

    # 2. Clean data
    df["Appointment_Date"] = pd.to_datetime(df["Appointment_Date"])
    df = df.drop_duplicates()
    df["Revenue"] = df["Revenue"].fillna(0)

    # 3. Create date fields
    df["Month"] = df["Appointment_Date"].dt.month_name()
    df["Month_Number"] = df["Appointment_Date"].dt.month
    df["Year"] = df["Appointment_Date"].dt.year

    # 4. Calculate KPIs
    total_appointments = len(df)
    completed = (df["Appointment_Status"] == "Completed").sum()
    cancelled = (df["Appointment_Status"] == "Cancelled").sum()
    no_show = (df["Appointment_Status"] == "No Show").sum()

    completion_rate = (
        completed / total_appointments * 100 if total_appointments else 0
    )
    no_show_rate = (
        no_show / total_appointments * 100 if total_appointments else 0
    )

    # Count revenue only for completed appointments
    total_revenue = df.loc[
        df["Appointment_Status"] == "Completed", "Revenue"
    ].sum()

    print("\n===== OVERALL KPIs =====")
    print(f"Total appointments : {total_appointments}")
    print(f"Completed          : {completed}")
    print(f"Cancelled          : {cancelled}")
    print(f"No-show            : {no_show}")
    print(f"Completion rate    : {completion_rate:.2f}%")
    print(f"No-show rate       : {no_show_rate:.2f}%")
    print(f"Completed revenue  : {total_revenue:,.2f}")

    # 5. Hospital-wise analysis
    hospital_analysis = (
        df.groupby("Hospital")
        .agg(
            Total_Appointments=("Appointment_ID", "count"),
            Completed=(
                "Appointment_Status",
                lambda values: (values == "Completed").sum(),
            ),
            Total_Revenue=(
                "Revenue",
                lambda values: values[
                    df.loc[values.index, "Appointment_Status"] == "Completed"
                ].sum(),
            ),
        )
        .reset_index()
    )

    print("\n===== HOSPITAL ANALYSIS =====")
    print(hospital_analysis.to_string(index=False))

    # 6. Provider-wise analysis
    provider_analysis = (
        df.groupby("Provider")
        .agg(
            Total_Appointments=("Appointment_ID", "count"),
            Total_Revenue=(
                "Revenue",
                lambda values: values[
                    df.loc[values.index, "Appointment_Status"] == "Completed"
                ].sum(),
            ),
        )
        .reset_index()
        .sort_values("Total_Appointments", ascending=False)
    )

    print("\n===== PROVIDER ANALYSIS =====")
    print(provider_analysis.to_string(index=False))

    # 7. Monthly revenue analysis (chronological order)
    completed_df = df[df["Appointment_Status"] == "Completed"].copy()
    monthly_revenue = (
        completed_df.groupby("Month_Number")["Revenue"]
        .sum()
        .reindex(range(1, 13), fill_value=0)
    )
    monthly_revenue.index = [
        pd.Timestamp(month=month, day=1, year=2026).strftime("%b")
        for month in monthly_revenue.index
    ]

    print("\n===== MONTHLY COMPLETED REVENUE =====")
    print(monthly_revenue.to_string())

    # 8. Appointment status chart
    status_counts = df["Appointment_Status"].value_counts()
    status_counts.plot(kind="bar", title="Appointment Status Analysis")
    plt.xlabel("Appointment Status")
    plt.ylabel("Number of Appointments")
    plt.tight_layout()
    plt.savefig("appointment_status.png", dpi=150)
    plt.close()

    # 9. Hospital revenue chart
    sns.barplot(
        data=hospital_analysis,
        x="Hospital",
        y="Total_Revenue",
    )
    plt.title("Hospital-wise Completed Revenue")
    plt.xlabel("Hospital")
    plt.ylabel("Completed Revenue")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig("hospital_revenue.png", dpi=150)
    plt.close()

    # 10. Export CSV files
    df.to_csv("healthcare_analysis.csv", index=False)
    hospital_analysis.to_csv("hospital_analysis.csv", index=False)
    provider_analysis.to_csv("provider_analysis.csv", index=False)
    monthly_revenue.rename("Revenue").to_csv("monthly_revenue.csv")

    # 11. Save dataset in SQLite database
    with sqlite3.connect("healthcare.db") as connection:
        df.to_sql("appointments", connection, if_exists="replace", index=False)

    print("\nAnalysis completed successfully.")
    print("Created CSV files, healthcare.db, and chart PNG files.")


if __name__ == "__main__":
    main()
