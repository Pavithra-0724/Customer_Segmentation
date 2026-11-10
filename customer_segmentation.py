
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Generate Customer Dataset
np.random.seed(42)

data = {
    "Customer_ID": range(1, 241),
    "Age": np.random.randint(18, 65, 240),
    "Gender": np.random.choice(["Male", "Female"], 240),
    "Annual_Income": np.concatenate([
        np.random.normal(25000, 5000, 60),
        np.random.normal(75000, 12000, 60),
        np.random.normal(45000, 8000, 60),
        np.random.normal(90000, 10000, 60)
    ]),
    "Spending_Score": np.concatenate([
        np.random.normal(25, 8, 60),
        np.random.normal(85, 8, 60),
        np.random.normal(45, 10, 60),
        np.random.normal(35, 10, 60)
    ]),
    "Purchase_Frequency": np.concatenate([
        np.random.normal(3, 1, 60),
        np.random.normal(15, 3, 60),
        np.random.normal(7, 2, 60),
        np.random.normal(4, 1, 60)
    ]),
    "Avg_Order_Value": np.concatenate([
        np.random.normal(500, 100, 60),
        np.random.normal(2500, 400, 60),
        np.random.normal(1000, 200, 60),
        np.random.normal(1800, 300, 60)
    ])
}

df = pd.DataFrame(data)

df["Annual_Income"] = df["Annual_Income"].clip(lower=10000).round(0)
df["Spending_Score"] = df["Spending_Score"].clip(1, 100).round(0)
df["Purchase_Frequency"] = df["Purchase_Frequency"].clip(1).round(0)
df["Avg_Order_Value"] = df["Avg_Order_Value"].clip(100).round(0)

df.to_csv("customer_data.csv", index=False)

print("Customer dataset created successfully!")
print(df.head())

# 2. Select Features
features = [
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency",
    "Avg_Order_Value"
]

X = df[features]

# 3. Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Customer_Segment"] = kmeans.fit_predict(X_scaled)

# 5. Analyze Customer Segments
summary = df.groupby("Customer_Segment")[features].mean().round(2)

print("\nCUSTOMER SEGMENT ANALYSIS")
print(summary)

# 6. Save Segmented Dataset
df.to_csv("segmented_customers.csv", index=False)
summary.to_csv("segment_summary.csv")

# 7. Visualization: Customer Segments
plt.figure(figsize=(10, 6))

plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Customer_Segment"],
    cmap="viridis",
    s=60
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using KMeans")
plt.colorbar(label="Customer Segment")
plt.grid(alpha=0.3)
plt.savefig("customer_segments.png", dpi=300)
plt.show()

# 8. Visualization: Segment Characteristics
summary[["Spending_Score", "Purchase_Frequency"]].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Customer Segment Characteristics")
plt.xlabel("Customer Segment")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("segment_analysis.png", dpi=300)
plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("Generated files:")
print("1. customer_data.csv")
print("2. segmented_customers.csv")
print("3. segment_summary.csv")
print("4. customer_segments.png")
print("5. segment_analysis.png")
