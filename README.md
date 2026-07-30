# Association rule mining system with contextual variables using python

## Project Overview

This project analyzes bakery transaction data to identify products that are frequently purchased together and understand how purchasing patterns vary across different time periods.

The **Apriori algorithm** is used to discover frequent itemsets and generate association rules from customer transaction data. The project also considers time-related information, such as morning, afternoon, and evening, to provide additional context for product purchasing behavior.

The main goal is to identify meaningful relationships between products while ensuring that time-related information is treated as contextual data rather than as a product.

## Objectives

* Analyze bakery customer transaction data
* Identify frequently purchased products
* Discover products that are commonly purchased together
* Generate frequent itemsets using the Apriori algorithm
* Generate association rules between products
* Evaluate association rules using support, confidence, and lift
* Analyze product purchasing patterns based on time periods
* Understand customer purchasing behavior using transaction data

## Dataset

The project uses a bakery transaction dataset containing more than **20,000 transaction records**.

The dataset includes information such as:

* Transaction ID
* Date
* Time
* Product/Item

During preprocessing, time-related information is extracted and categorized into periods such as:

* Morning
* Afternoon
* Evening

The cleaned dataset is saved as:

```text id="4jv5a7"
bakery_cleaned.csv
```

## Technologies Used

* Python 3.11.9
* Pandas
* NumPy
* Matplotlib
* Mlxtend

## Evaluation Metrics

The generated association rules are evaluated using the following metrics.

### Support

Support indicates how frequently a product combination occurs in the complete dataset.

```text id="8qz2ks"
Support(A → B) = Transactions containing A and B / Total transactions
```

### Confidence

Confidence indicates how often product B is purchased when product A is purchased.

```text id="3ybqrw"
Confidence(A → B) = Support(A and B) / Support(A)
```

### Lift

Lift measures the strength of the relationship between two products.

```text id="1c6n9t"
Lift(A → B) = Confidence(A → B) / Support(B)
```

**Interpretation:**

* **Lift > 1:** Positive association
* **Lift = 1:** No significant association
* **Lift < 1:** Negative association

## Expected Results

The project provides:

* Frequently purchased bakery products
* Product combinations that commonly occur together
* Association rules between products
* Support, confidence, and lift values
* Insights into customer purchasing behavior

## Challenges

Some challenges encountered during the project include:

* Cleaning a large transaction dataset
* Converting transaction data into a format suitable for Apriori
* Handling missing or inconsistent values
* Selecting suitable minimum support and confidence values
* Preventing time-related variables from being incorrectly treated as products
* Filtering meaningful association rules from a large number of generated rules

## Future Enhancements

The project can be improved by:

* Comparing Apriori with FP-Growth
* Developing an interactive dashboard
* Applying recommendation techniques
* Using machine-learning models to predict future product demand

## Conclusion

This project demonstrates the use of the Apriori algorithm to identify meaningful product associations from bakery transaction data.

By separating actual products from time-related contextual information, the project produces more relevant and interpretable association rules. The results can help understand customer purchasing behavior, identify commonly purchased product combinations, and support data-driven business decisions.

## Author

**Divya Dharshana J**
(M.Sc. Computer Science)

## License

This project is created for educational and academic purposes.
