## Credit Scoring Business Understanding

This section provides a concise overview of key concepts in credit risk, informed by regulatory frameworks and practical modeling considerations.

---

### Why Basel II Demands Interpretable, Documented Models

Basel II requires banks to use internal risk estimates—**Probability of Default (PD)**, **Loss Given Default (LGD)**, and **Exposure at Default (EAD)**—which drives the need for transparent and reliable models.

- **Regulatory Compliance**: Models must demonstrate conceptual soundness and reliability to regulators.
- **Interpretability is Key**: Enables understanding of variable influence, model validation, bias detection, and regulatory alignment.
- **Documentation is Essential**: Ensures transparency across the model lifecycle (data, features, development, validation, monitoring), critical for:
  - Auditability  
  - Reproducibility  
  - Reducing model risk
- **What Happens Without It?**
  - Difficulty justifying capital allocations  
  - Poor risk management  
  - Regulatory non-compliance  

---

### Why We Use Proxy Variables (and Their Risks)

A direct "default" label is often unavailable, requiring the use of **proxy variables** (e.g., 90+ Days Past Due, charge-offs) to define the model target.

#### Why It's Necessary:
- Default is rarely clearly defined or standardized across data sources.
- Proxies provide a measurable target for supervised learning.

#### Business Risks of Using Proxies:
- **Misclassification**: Proxy ≠ actual default → mislabels in training data.
- **Inaccurate Risk Scoring**: Poor PD estimates may affect:
  - Pricing  
  - Provisioning  
  - Capital planning
- **Suboptimal Lending**:
  - *False Positives*: Rejecting good borrowers → lost revenue.
  - *False Negatives*: Approving bad borrowers → increased loss.
- **Regulatory Risk**: Weak justifications for proxies = scrutiny or penalties.
- **Model Drift**: Over time, the proxy may no longer reflect actual default behavior.

---

### Interpretable vs. Complex Models: The Regulatory Trade-Off

Choosing between model types in credit risk is a trade-off between **accuracy** and **regulatory compliance**.

#### Interpretable Models (e.g., Logistic Regression + WoE)

**Pros:**
- Clear, explainable predictions (great for customer transparency and regulator trust)
- Easier to audit and validate
- Generally more stable and less prone to overfitting

**Cons:**
- Limited predictive power on complex data
- Often requires extensive manual feature engineering

#### Complex Models (e.g., Gradient Boosting Machines)

**Pros:**
- Superior predictive accuracy
- Handles non-linear relationships and feature interactions well
- Less need for manual feature preprocessing

**Cons:**
- Harder to explain (black-box behavior)
- Challenging to audit and justify
- Greater risk of overfitting
- May face regulatory pushback due to low transparency

---

### Practical Approach: Balance and Hybrid Strategies

- **In Practice**: Simpler models are often preferred in core risk areas (e.g., credit approval, provisioning) due to interpretability.
- **Hybrid Strategy**:
  - Use complex models for risk scoring
  - Apply interpretability layers (e.g., SHAP, LIME) post-hoc
  - Combine with simple benchmark models for validation
