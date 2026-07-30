# =====================================================
# CONTEXT-AWARE ASSOCIATION RULE MINING
# Bakery Dataset
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(
    "C:/Users/divya/OneDrive/Desktop/Project/bakery_cleaned.csv"
)

print("\nDataset Shape:", df.shape)

# =====================================================
# FUNCTION TO RUN APRIORI
# =====================================================

def run_apriori(data,
                context_name,
                min_support=0.01,
                min_confidence=0.30,
                top_k=10):

    print("\n")
    print("="*80)
    print("CONTEXT :", context_name)
    print("="*80)

    transactions = []

    for tid, group in data.groupby('transaction'):

        items = list(group['item'].unique())

        if len(items) >= 2:
            transactions.append(items)

    if len(transactions) == 0:
        return None

    print("Transactions:", len(transactions))

    # -----------------------------------------
    # One Hot Encoding
    # -----------------------------------------

    te = TransactionEncoder()

    basket = te.fit(transactions).transform(transactions)

    basket = pd.DataFrame(
        basket,
        columns=te.columns_
    )

    # -----------------------------------------
    # Apriori
    # -----------------------------------------

    frequent = apriori(
        basket,
        min_support=min_support,
        use_colnames=True
    )

    if len(frequent) == 0:
        return None

    rules = association_rules(
        frequent,
        metric='confidence',
        min_threshold=min_confidence
    )

    if len(rules) == 0:
        return None

    # -----------------------------------------
    # Clean output
    # -----------------------------------------

    rules = rules[
        [
            'antecedents',
            'consequents',
            'support',
            'confidence',
            'lift'
        ]
    ]

    rules = rules[
        rules['lift'] > 1
    ]

    rules = rules.sort_values(
        'lift',
        ascending=False
    )

    rules['antecedents'] = rules[
        'antecedents'
    ].apply(
        lambda x:
        ", ".join(list(x))
    )

    rules['consequents'] = rules[
        'consequents'
    ].apply(
        lambda x:
        ", ".join(list(x))
    )

    rules['rule'] = (
        rules['antecedents']
        + " → "
        + rules['consequents']
    )

    rules = rules.round(3)

    # -----------------------------------------
    # TOP 5 RULES BY HIGHEST CONFIDENCE
    # -----------------------------------------

    top_rules = (
        rules
        .sort_values(
            by='confidence',
            ascending=False
        )
        .head(5)
    )

    print("\n")
    print("="*80)
    print("TOP 5 ASSOCIATION RULES BY CONFIDENCE")
    print("="*80)

    for rank, (_, row) in enumerate(
            top_rules.iterrows(),
            start=1):

        print(f"\nRank {rank}")

        print(
            f"Rule: {row['rule']}"
        )

        print(
            f"Confidence: "
            f"{row['confidence']*100:.2f}%"
        )

        print(
            f"Support: "
            f"{row['support']*100:.2f}%"
        )

        print(
            f"Lift: "
            f"{row['lift']}"
        )

        print(
            f"Explanation:"
        )

        print(
            f"Among customers who buy "
            f"{row['antecedents']}, "
            f"{row['confidence']*100:.2f}% "
            f"also purchase "
            f"{row['consequents']}."
        )

        print("-"*60)

    # -----------------------------------------
    # Chart
    # -----------------------------------------

    top = rules.head(10)

    plt.figure(figsize=(10,6))

    plt.bar(
        top['rule'],
        top['lift']
    )

    plt.xticks(rotation=90)

    plt.title(
        f"Top Rules : {context_name}"
    )

    plt.xlabel("Association Rule")
    plt.ylabel("Lift")

    plt.tight_layout()

    plt.show()

    return rules


# =====================================================
# OVERALL DATASET
# =====================================================

overall_rules = run_apriori(
    df,
    "Overall Dataset"
)

# =====================================================
# PERIOD ANALYSIS
# =====================================================

for period in df['period_day'].unique():

    subset = df[
        df['period_day'] == period
    ]

    run_apriori(
        subset,
        f"Period : {period}"
    )

# =====================================================
# WEEKDAY/WEEKEND ANALYSIS
# =====================================================

for day in df[
    'weekday_weekend'
].unique():

    subset = df[
        df['weekday_weekend'] == day
    ]

    run_apriori(
        subset,
        f"Day Type : {day}"
    )

# =====================================================
# COMBINED CONTEXT
# =====================================================

contexts = (
    df.groupby(
        [
            'period_day',
            'weekday_weekend'
        ]
    )
)

for context, subset in contexts:

    period = context[0]
    day = context[1]

    run_apriori(
        subset,
        f"{period} + {day}"
    )
