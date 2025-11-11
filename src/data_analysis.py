import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_datasets():
    """load both csv files"""
    paper_df = pd.read_csv('data/processed/ret_multivariant_training_data.csv')
    expanded_df = pd.read_csv('data/processed/ret_multivariant_expanded_training_data.csv')
    return paper_df, expanded_df


def print_descriptive_statistics(paper_df, expanded_df):
    """provide descriptive statistics for key features"""
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    key_features = ['age', 'calcitonin_level_numeric', 'thyroid_nodules_present', 'multiple_nodules']

    for feature in key_features:
        print(f"\n{feature.upper()} STATISTICS:")
        print(f"Paper dataset: mean={paper_df[feature].mean():.2f}, std={paper_df[feature].std():.2f}")
        print(f"Expanded dataset: mean={expanded_df[feature].mean():.2f}, std={expanded_df[feature].std():.2f}")

    print("\nMEN2 DIAGNOSIS BREAKDOWN:")
    print("Paper dataset:")
    print(paper_df['men2_syndrome'].value_counts())
    print("\nExpanded dataset:")
    print(expanded_df['men2_syndrome'].value_counts())

    print("\nRET VARIANT DISTRIBUTION:")
    print("Paper dataset:")
    variant_counts = paper_df['ret_variant'].value_counts()
    for variant, count in variant_counts.items():
        risk_level = paper_df[paper_df['ret_variant'] == variant]['ret_risk_level'].iloc[0]
        mtc_rate = paper_df[paper_df['ret_variant'] == variant]['mtc_diagnosis'].mean()
        print(f"  {variant} (Risk Level {risk_level}): {count} patients, MTC rate: {mtc_rate:.1%}")
    print("\nExpanded dataset:")
    variant_counts_exp = expanded_df['ret_variant'].value_counts()
    for variant, count in variant_counts_exp.items():
        risk_level = expanded_df[expanded_df['ret_variant'] == variant]['ret_risk_level'].iloc[0]
        mtc_rate = expanded_df[expanded_df['ret_variant'] == variant]['mtc_diagnosis'].mean()
        print(f"  {variant} (Risk Level {risk_level}): {count} patients, MTC rate: {mtc_rate:.1%}")


def generate_plots(paper_df, expanded_df):
    """generate and display informative plots"""
    # set style
    plt.style.use('seaborn-v0_8')
    os.makedirs('charts', exist_ok=True)

    # 1. Histograms of age for cases vs controls
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # paper dataset age distribution
    paper_df[paper_df['mtc_diagnosis'] == 0]['age'].hist(ax=ax1, alpha=0.7, label='No MTC', bins=10)
    paper_df[paper_df['mtc_diagnosis'] == 1]['age'].hist(ax=ax1, alpha=0.7, label='MTC', bins=10)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Paper Dataset: Age Distribution by MTC Diagnosis')
    ax1.legend()

    # expanded dataset age distribution
    expanded_df[expanded_df['mtc_diagnosis'] == 0]['age'].hist(ax=ax2, alpha=0.7, label='No MTC', bins=15)
    expanded_df[expanded_df['mtc_diagnosis'] == 1]['age'].hist(ax=ax2, alpha=0.7, label='MTC', bins=15)
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Expanded Dataset: Age Distribution by MTC Diagnosis')
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join('charts', 'age_histograms.png'), dpi=200)
    plt.close(fig)

    # 2. Boxplots of calcitonin levels by MTC and MEN2 diagnosis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # calcitonin by MTC diagnosis
    mtc_groups = [paper_df[paper_df['mtc_diagnosis'] == 0]['calcitonin_level_numeric'],
                  paper_df[paper_df['mtc_diagnosis'] == 1]['calcitonin_level_numeric']]
    ax1.boxplot(mtc_groups, tick_labels=['No MTC', 'MTC'])
    ax1.set_ylabel('Calcitonin Level (pg/mL)')
    ax1.set_title('Paper Dataset: Calcitonin by MTC Diagnosis')

    # calcitonin by MEN2 diagnosis
    men2_groups = [paper_df[paper_df['men2_syndrome'] == 0]['calcitonin_level_numeric'],
                   paper_df[paper_df['men2_syndrome'] == 1]['calcitonin_level_numeric']]
    ax2.boxplot(men2_groups, tick_labels=['No MEN2', 'MEN2'])
    ax2.set_ylabel('Calcitonin Level (pg/mL)')
    ax2.set_title('Paper Dataset: Calcitonin by MEN2 Diagnosis')

    plt.tight_layout()
    fig.savefig(os.path.join('charts', 'calcitonin_boxplots.png'), dpi=200)
    plt.close(fig)

    # 3. Bar charts of feature distributions by diagnosis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    # nodule presence by MTC diagnosis
    nodule_mtc = pd.crosstab(paper_df['mtc_diagnosis'], paper_df['thyroid_nodules_present'])
    nodule_mtc.plot(kind='bar', ax=axes[0], stacked=True)
    axes[0].set_title('Thyroid Nodules by MTC Diagnosis')
    axes[0].set_xlabel('MTC Diagnosis')
    axes[0].set_ylabel('Count')
    axes[0].legend(['No Nodules', 'Nodules Present'])

    # calcitonin elevated by MTC diagnosis
    calcitonin_mtc = pd.crosstab(paper_df['mtc_diagnosis'], paper_df['calcitonin_elevated'])
    calcitonin_mtc.plot(kind='bar', ax=axes[1], stacked=True)
    axes[1].set_title('Calcitonin Elevated by MTC Diagnosis')
    axes[1].set_xlabel('MTC Diagnosis')
    axes[1].set_ylabel('Count')
    axes[1].legend(['Normal', 'Elevated'])

    # gender distribution by MTC diagnosis
    gender_mtc = pd.crosstab(paper_df['mtc_diagnosis'], paper_df['gender'])
    gender_mtc.plot(kind='bar', ax=axes[2], stacked=True)
    axes[2].set_title('Gender by MTC Diagnosis')
    axes[2].set_xlabel('MTC Diagnosis')
    axes[2].set_ylabel('Count')
    axes[2].legend(['Female', 'Male'])

    # age group by MTC diagnosis
    age_mtc = pd.crosstab(paper_df['mtc_diagnosis'], paper_df['age_group'])
    age_mtc.plot(kind='bar', ax=axes[3], stacked=True)
    axes[3].set_title('Age Group by MTC Diagnosis')
    axes[3].set_xlabel('MTC Diagnosis')
    axes[3].set_ylabel('Count')

    plt.tight_layout()
    fig.savefig(os.path.join('charts', 'feature_distributions.png'), dpi=200)
    plt.close(fig)

    # 4. RET Variant Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Variant counts
    variant_counts = paper_df['ret_variant'].value_counts()
    colors = sns.color_palette("husl", len(variant_counts))
    ax1.bar(range(len(variant_counts)), variant_counts.values, color=colors)
    ax1.set_xticks(range(len(variant_counts)))
    ax1.set_xticklabels(variant_counts.index, rotation=45, ha='right')
    ax1.set_ylabel('Patient Count')
    ax1.set_title('RET Variant Distribution (Paper Dataset)')
    ax1.grid(axis='y', alpha=0.3)

    # MTC diagnosis rate by variant
    mtc_by_variant = paper_df.groupby('ret_variant')['mtc_diagnosis'].agg(['mean', 'count'])
    ax2.bar(range(len(mtc_by_variant)), mtc_by_variant['mean'].values, color=colors)
    ax2.set_xticks(range(len(mtc_by_variant)))
    ax2.set_xticklabels(mtc_by_variant.index, rotation=45, ha='right')
    ax2.set_ylabel('MTC Diagnosis Rate')
    ax2.set_title('MTC Diagnosis Rate by RET Variant')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.0])

    plt.tight_layout()
    fig.savefig(os.path.join('charts', 'variant_distribution.png'), dpi=200)
    plt.close(fig)

    # 5. Calcitonin levels by variant
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for boxplot
    variants = paper_df['ret_variant'].unique()
    data_by_variant = [paper_df[paper_df['ret_variant'] == v]['calcitonin_level_numeric'].values for v in variants]

    bp = ax.boxplot(data_by_variant, labels=variants, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(variants)]):
        patch.set_facecolor(color)

    ax.set_ylabel('Calcitonin Level (pg/mL)')
    ax.set_xlabel('RET Variant')
    ax.set_title('Calcitonin Levels by RET Variant')
    ax.tick_params(axis='x', rotation=45)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join('charts', 'calcitonin_by_variant.png'), dpi=200)
    plt.close(fig)

    # 6. Risk level analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # MTC rate by risk level
    mtc_by_risk = paper_df.groupby('ret_risk_level')['mtc_diagnosis'].mean()
    risk_labels = {1: 'Level 1\n(Moderate)', 2: 'Level 2\n(High)', 3: 'Level 3\n(Highest)'}
    ax1.bar([risk_labels[k] for k in mtc_by_risk.index], mtc_by_risk.values, color=['green', 'orange', 'red'][:len(mtc_by_risk)])
    ax1.set_ylabel('MTC Diagnosis Rate')
    ax1.set_xlabel('ATA Risk Level')
    ax1.set_title('MTC Diagnosis Rate by ATA Risk Level')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.0])

    # Patient count by risk level
    risk_counts = paper_df['ret_risk_level'].value_counts().sort_index()
    ax2.bar([risk_labels[k] for k in risk_counts.index], risk_counts.values, color=['green', 'orange', 'red'][:len(risk_counts)])
    ax2.set_ylabel('Patient Count')
    ax2.set_xlabel('ATA Risk Level')
    ax2.set_title('Patient Distribution by ATA Risk Level')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join('charts', 'risk_level_analysis.png'), dpi=200)
    plt.close(fig)


def print_insights(paper_df, expanded_df):
    """print insights regarding data distributions and class imbalance"""
    print("=" * 60)
    print("DATA ANALYSIS INSIGHTS")
    print("=" * 60)

    print("PAPER DATASET INSIGHTS:")
    print(f"- Total patients: {len(paper_df)}")
    print(f"- MTC cases: {paper_df['mtc_diagnosis'].sum()}/4 ({paper_df['mtc_diagnosis'].mean():.1%})")
    print(f"- C-cell disease cases: {paper_df['c_cell_disease'].sum()}/4 ({paper_df['c_cell_disease'].mean():.1%})")
    print(f"- MEN2 syndrome cases: {paper_df['men2_syndrome'].sum()}/4 ({paper_df['men2_syndrome'].mean():.1%})")
    print(f"- Average age: {paper_df['age'].mean():.1f} years")
    print(f"- Gender distribution: {paper_df['gender'].value_counts().to_dict()}")
    print(f"- Calcitonin elevated: {paper_df['calcitonin_elevated'].sum()}/4 ({paper_df['calcitonin_elevated'].mean():.1%})")
    print(f"- Thyroid nodules present: {paper_df['thyroid_nodules_present'].sum()}/4 ({paper_df['thyroid_nodules_present'].mean():.1%})")
    print()

    print("EXPANDED DATASET INSIGHTS:")
    print(f"- Total patients: {len(expanded_df)}")
    print(f"- MTC cases: {expanded_df['mtc_diagnosis'].sum()}/{len(expanded_df)} ({expanded_df['mtc_diagnosis'].mean():.1%})")
    print(f"- C-cell disease cases: {expanded_df['c_cell_disease'].sum()}/{len(expanded_df)} ({expanded_df['c_cell_disease'].mean():.1%})")
    print(f"- MEN2 syndrome cases: {expanded_df['men2_syndrome'].sum()}/{len(expanded_df)} ({expanded_df['men2_syndrome'].mean():.1%})")
    print(f"- Average age: {expanded_df['age'].mean():.1f} years")
    print(f"- Gender distribution: {expanded_df['gender'].value_counts().to_dict()}")
    print(f"- Calcitonin elevated: {expanded_df['calcitonin_elevated'].sum()}/{len(expanded_df)} ({expanded_df['calcitonin_elevated'].mean():.1%})")
    print(f"- Thyroid nodules present: {expanded_df['thyroid_nodules_present'].sum()}/{len(expanded_df)} ({expanded_df['thyroid_nodules_present'].mean():.1%})")
    print()

    print("RET VARIANT INSIGHTS:")
    variant_stats = paper_df.groupby('ret_variant').agg({
        'mtc_diagnosis': ['count', 'sum', 'mean'],
        'ret_risk_level': 'first'
    })
    variant_stats.columns = ['_'.join(col).strip() for col in variant_stats.columns.values]
    for variant in variant_stats.index:
        count = int(variant_stats.loc[variant, 'mtc_diagnosis_count'])
        mtc_count = int(variant_stats.loc[variant, 'mtc_diagnosis_sum'])
        mtc_rate = variant_stats.loc[variant, 'mtc_diagnosis_mean']
        risk_level = int(variant_stats.loc[variant, 'ret_risk_level_first'])
        print(f"- {variant} (Risk Level {risk_level}): {count} patients, {mtc_count} with MTC ({mtc_rate:.1%})")
    print()

    print("RISK LEVEL INSIGHTS:")
    risk_stats = paper_df.groupby('ret_risk_level').agg({
        'mtc_diagnosis': ['count', 'sum', 'mean']
    })
    risk_stats.columns = ['_'.join(col).strip() for col in risk_stats.columns.values]
    risk_labels = {1: 'Moderate', 2: 'High', 3: 'Highest'}
    for risk_level in risk_stats.index:
        count = int(risk_stats.loc[risk_level, 'mtc_diagnosis_count'])
        mtc_count = int(risk_stats.loc[risk_level, 'mtc_diagnosis_sum'])
        mtc_rate = risk_stats.loc[risk_level, 'mtc_diagnosis_mean']
        label = risk_labels.get(risk_level, str(risk_level))
        print(f"- Level {risk_level} ({label}): {count} patients, {mtc_count} with MTC ({mtc_rate:.1%})")
    print()

    print("CLASS IMBALANCE OBSERVATIONS:")
    print("- MTC diagnosis shows significant class imbalance (25% in paper, ~50% in expanded)")
    print("- C-cell disease is more balanced but still shows some imbalance")
    print("- MEN2 syndrome is extremely imbalanced (0% in paper dataset)")
    print("- Age and calcitonin levels show clear differentiation between MTC+ and MTC- groups")
    print("- Gender distribution appears relatively balanced")
    print("- Thyroid nodules strongly associated with MTC diagnosis")
    print("- RET variant distribution includes K666N (moderate risk) and multiple other variants")
    print("- Higher risk level variants (C634*) show different MTC penetrance patterns")
    print("=" * 60)


if __name__ == "__main__":
    paper_df, expanded_df = load_datasets()
    print_descriptive_statistics(paper_df, expanded_df)
    generate_plots(paper_df, expanded_df)
    print_insights(paper_df, expanded_df)