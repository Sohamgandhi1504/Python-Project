# MARKET BASKET ANALYSIS PROJECT - FINAL VERSION
# ----------------------------------------------
# Author: Pulkit Jain
# Description: Runs Market Basket Analysis on Groceries dataset
# using Apriori algorithm and generates clean visual + CSV outputs.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ------------------------------
# Step 1 — Load Dataset
# ------------------------------
print("📦 Loading dataset...")
df = pd.read_csv(r"C:\testdata\groceries_clean.csv", encoding="latin1", engine="python")
print("✅ Dataset loaded successfully!")
print(df.head())

# ------------------------------
# Step 2 — Data Cleaning
# ------------------------------
print("\n🧹 Data Cleaning Started...")

df.columns = ["Member_number", "Date", "itemDescription"]
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df = df.dropna()

# Group transactions by Member_number and Date
transactions = (
    df.groupby(["Member_number", "Date"])["itemDescription"]
      .apply(list)
      .reset_index()
)

print("✅ Data cleaned and transactions grouped successfully!")
print(transactions.head())

# ------------------------------
# Step 3 — Encoding Transactions
# ------------------------------
print("\n🔢 Encoding transactions for Apriori...")

basket_lists = transactions["itemDescription"].tolist()
te = TransactionEncoder()
te_array = te.fit(basket_lists).transform(basket_lists)
basket_df = pd.DataFrame(te_array, columns=te.columns_)

print("✅ Encoding complete! Shape:", basket_df.shape)

# ------------------------------
# Step 4 — Apriori Algorithm
# ------------------------------
print("\n⚙️ Running Apriori algorithm...")

frequent_itemsets = apriori(basket_df, min_support=0.005, use_colnames=True)
print("\n--- Frequent Itemsets ---")
print(frequent_itemsets.head())

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values(by="lift", ascending=False)

print("\n✅ Apriori algorithm completed successfully!")
print("\n--- Top 10 Association Rules ---")
print(rules.head(10)[["antecedents", "consequents", "support", "confidence", "lift"]])

# ------------------------------
# Step 5 — Fix frozensets BEFORE saving
# ------------------------------
print("\n🧼 Cleaning frozenset columns...")

def frozenset_to_str(x):
    """Convert frozenset({'milk','bread'}) → 'milk, bread'"""
    if isinstance(x, frozenset):
        return ', '.join(sorted(list(x)))
    elif isinstance(x, str):
        return x.replace("frozenset(", "").replace("{", "").replace("}", "").replace("'", "").replace(")", "")
    else:
        return str(x)

rules['antecedents'] = rules['antecedents'].apply(frozenset_to_str)
rules['consequents'] = rules['consequents'].apply(frozenset_to_str)
rules['support'] = rules['support'].round(3)
rules['confidence'] = rules['confidence'].round(3)
rules['lift'] = rules['lift'].round(3)

print("✅ Frozenset issue fixed successfully!")

# ------------------------------
# Step 6 — Visualization
# ------------------------------
print("\n📊 Generating visualizations...")

# Top 10 frequent itemsets
top_items = frequent_itemsets.sort_values(by="support", ascending=False).head(10)
top_items["itemsets"] = top_items["itemsets"].apply(lambda x: ', '.join(list(x)))

plt.figure(figsize=(10,5))
sns.barplot(x="support", y="itemsets", data=top_items, color="skyblue")
plt.title("Top 10 Frequent Itemsets")
plt.xlabel("Support")
plt.ylabel("Items")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/top_items.png")
plt.close()

# Top 10 association rules by lift
top_rules = rules.sort_values(by="lift", ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x="lift", y="consequents", data=top_rules, color="lightgreen")
plt.title("Top 10 Association Rules by Lift")
plt.xlabel("Lift")
plt.ylabel("Consequents")
plt.tight_layout()
plt.savefig("outputs/top_rules.png")
plt.close()

print("✅ Visualizations saved in 'outputs' folder!")

# ------------------------------
# Step 7 — Save Results
# ------------------------------
print("\n💾 Saving CSV outputs...")

os.makedirs("outputs", exist_ok=True)
frequent_itemsets.to_csv("outputs/frequent_itemsets.csv", index=False)
rules.to_csv("outputs/association_rules.csv", index=False, encoding="utf-8-sig")

print("✅ Files saved successfully in 'outputs' folder!")

# ------------------------------
# Step 8 — Display Summary
# ------------------------------
print("\n📄 --- Top 5 Rules (Readable Format) ---")
for i, row in rules.head(5).iterrows():
    print(f"Rule {i+1}: If a customer buys [{row['antecedents']}], they’re likely to also buy [{row['consequents']}]")
    print(f"  - Support: {row['support']}")
    print(f"  - Confidence: {row['confidence']}")
    print(f"  - Lift: {row['lift']}")
    print()

print("🏁 All steps completed successfully!")
