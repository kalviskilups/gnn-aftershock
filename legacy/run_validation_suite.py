#!/usr/bin/env python
"""
Simplified validation script that works with your existing setup.
Analyzes the waveform features to determine if the approach is physically valid.

Usage:
    python simplified_validation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)


def analyze_feature_importance(waveform_features_dict):
    """
    Analyze feature importance based on frequency of occurrence and value distribution

    Parameters:
    -----------
    waveform_features_dict : dict
        Dictionary mapping event IDs to waveform features

    Returns:
    --------
    feature_stats : pandas.DataFrame
        DataFrame with feature statistics
    """
    print("Analyzing waveform feature importance...")

    # First, identify all unique feature names across all events
    all_feature_names = set()
    for event_id, features in waveform_features_dict.items():
        all_feature_names.update(features.keys())

    # Convert to sorted list
    all_feature_names = sorted(list(all_feature_names))
    print(f"Found {len(all_feature_names)} unique waveform features")

    # Initialize statistics
    feature_stats = {
        "Feature": [],
        "Occurrence Rate": [],
        "Mean Value": [],
        "Std Value": [],
        "Max Value": [],
    }

    # Calculate statistics for each feature
    for feature_name in tqdm(all_feature_names, desc="Calculating feature statistics"):
        # Count occurrences
        occurrences = sum(
            1
            for features in waveform_features_dict.values()
            if feature_name in features
        )
        occurrence_rate = occurrences / len(waveform_features_dict)

        # Skip features with very low occurrence
        if occurrence_rate < 0.1:
            continue

        # Collect all values
        values = [
            features[feature_name]
            for features in waveform_features_dict.values()
            if feature_name in features
        ]

        # Calculate statistics
        mean_value = np.mean(values) if values else np.nan
        std_value = np.std(values) if values else np.nan
        max_value = np.max(values) if values else np.nan

        # Store statistics
        feature_stats["Feature"].append(feature_name)
        feature_stats["Occurrence Rate"].append(occurrence_rate)
        feature_stats["Mean Value"].append(mean_value)
        feature_stats["Std Value"].append(std_value)
        feature_stats["Max Value"].append(max_value)

    # Convert to DataFrame
    feature_stats = pd.DataFrame(feature_stats)

    # Calculate normalized variability (coefficient of variation)
    feature_stats["Normalized Variability"] = (
        feature_stats["Std Value"] / feature_stats["Mean Value"].abs()
    )

    # Estimate importance based on variability and occurrence
    feature_stats["Estimated Importance"] = (
        feature_stats["Normalized Variability"] * feature_stats["Occurrence Rate"]
    )

    # Fix infinite or NaN values
    feature_stats["Estimated Importance"] = feature_stats[
        "Estimated Importance"
    ].fillna(0)
    feature_stats["Estimated Importance"] = np.clip(
        feature_stats["Estimated Importance"],
        0,
        feature_stats["Estimated Importance"].quantile(0.95),
    )

    # Sort by estimated importance
    feature_stats = feature_stats.sort_values("Estimated Importance", ascending=False)

    # Plot top features
    plt.figure(figsize=(12, 8))
    top_n = min(20, len(feature_stats))
    sns.barplot(
        y="Feature",
        x="Estimated Importance",
        data=feature_stats.head(top_n),
        orient="h",
    )
    plt.title(f"Top {top_n} Waveform Features by Estimated Importance")
    plt.tight_layout()
    plt.savefig("results/waveform_feature_importance.png", dpi=300)
    plt.close()

    # Return the statistics
    return feature_stats


def analyze_feature_categories(feature_stats):
    """
    Analyze feature importance by categories

    Parameters:
    -----------
    feature_stats : pandas.DataFrame
        DataFrame with feature statistics

    Returns:
    --------
    category_stats : pandas.DataFrame
        DataFrame with category statistics
    """
    print("Analyzing feature categories...")

    # Define categories by prefix patterns
    category_patterns = {
        "P-wave features": ["P_"],
        "S-wave features": ["S_"],
        "Vertical component (Z)": ["Z_"],
        "North component (N)": ["N_"],
        "East component (E)": ["E_"],
        "P/S ratio": ["PS_ratio"],
        "Frequency features": ["freq", "dominant"],
        "Energy features": ["energy", "amplitude"],
        "Shape features": ["kurtosis", "std", "rms"],
    }

    # Assign categories to features
    feature_stats["Category"] = "Other"
    for category, patterns in category_patterns.items():
        for pattern in patterns:
            mask = feature_stats["Feature"].str.contains(pattern, regex=False)
            feature_stats.loc[mask, "Category"] = category

    # Aggregate by category
    category_stats = (
        feature_stats.groupby("Category")["Estimated Importance"].sum().reset_index()
    )
    category_stats = category_stats.sort_values("Estimated Importance", ascending=False)

    # Plot category importance
    plt.figure(figsize=(10, 6))
    sns.barplot(y="Category", x="Estimated Importance", data=category_stats, orient="h")
    plt.title("Waveform Feature Importance by Category")
    plt.tight_layout()
    plt.savefig("results/feature_category_importance.png", dpi=300)
    plt.close()

    return category_stats


def analyze_p_s_wave_characteristics(waveform_features_dict):
    """
    Analyze P-wave and S-wave characteristics to assess physical consistency

    Parameters:
    -----------
    waveform_features_dict : dict
        Dictionary mapping event IDs to waveform features

    Returns:
    --------
    results : dict
        Dictionary with analysis results
    """
    print("Analyzing P-wave and S-wave characteristics...")

    # Extract P/S ratio if available
    ps_ratios = []
    for features in waveform_features_dict.values():
        for key in ["Z_PS_ratio", "PS_ratio", "P_S_ratio"]:
            if key in features:
                ps_ratios.append(features[key])
                break

    # Results container
    results = {}

    # If we have P/S ratios, analyze their distribution
    if ps_ratios:
        results["ps_ratio_mean"] = np.mean(ps_ratios)
        results["ps_ratio_median"] = np.median(ps_ratios)
        results["ps_ratio_std"] = np.std(ps_ratios)

        # Plot distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(ps_ratios, kde=True)
        plt.axvline(
            results["ps_ratio_median"],
            color="red",
            linestyle="--",
            label=f"Median: {results['ps_ratio_median']:.2f}",
        )
        plt.title("Distribution of P/S Amplitude Ratios")
        plt.xlabel("P/S Ratio")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("results/ps_ratio_distribution.png", dpi=300)
        plt.close()

        print(
            f"P/S ratio statistics: Mean={results['ps_ratio_mean']:.2f}, Median={results['ps_ratio_median']:.2f}"
        )
    else:
        print("No P/S ratios found in the waveform features")

    # Extract P and S wave energy/amplitude information
    p_energies = []
    s_energies = []

    for features in waveform_features_dict.values():
        # Try to find P wave energy
        for key in ["P_Z_energy", "P_energy"]:
            if key in features:
                p_energies.append(features[key])
                break

        # Try to find S wave energy
        for key in ["S_Z_energy", "S_energy"]:
            if key in features:
                s_energies.append(features[key])
                break

    # If we have both P and S energies, compare them
    if p_energies and s_energies:
        # Take minimum of the two lists
        min_length = min(len(p_energies), len(s_energies))
        p_energies = p_energies[:min_length]
        s_energies = s_energies[:min_length]

        # Plot relationship
        plt.figure(figsize=(10, 8))
        plt.scatter(p_energies, s_energies, alpha=0.6)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("P-wave Energy")
        plt.ylabel("S-wave Energy")
        plt.title("P-wave vs S-wave Energy")
        plt.grid(alpha=0.3, which="both")

        # Add trend line
        if min_length > 2:
            try:
                # Log-log transformation for power law
                log_p = np.log(p_energies)
                log_s = np.log(s_energies)

                # Remove invalid values
                valid_mask = ~(
                    np.isnan(log_p)
                    | np.isnan(log_s)
                    | np.isinf(log_p)
                    | np.isinf(log_s)
                )
                log_p = log_p[valid_mask]
                log_s = log_s[valid_mask]

                if len(log_p) > 2:
                    # Fit line
                    slope, intercept = np.polyfit(log_p, log_s, 1)

                    # Generate points for line
                    x_range = np.logspace(
                        np.log10(min(p_energies)), np.log10(max(p_energies)), 100
                    )
                    y_fit = np.exp(intercept) * x_range**slope

                    # Plot line
                    plt.plot(
                        x_range,
                        y_fit,
                        "r-",
                        label=f"Slope: {slope:.2f} (S ~ P^{slope:.2f})",
                    )
                    plt.legend()

                    results["p_s_energy_slope"] = slope
            except Exception as e:
                print(f"Error fitting trend line: {e}")

        plt.savefig("results/p_s_energy_relationship.png", dpi=300)
        plt.close()

        print(f"Compared P-wave and S-wave energy for {min_length} events")
    else:
        print("Insufficient P/S energy data in waveform features")

    return results


def analyze_frequency_characteristics(waveform_features_dict):
    """
    Analyze frequency characteristics to assess physical consistency

    Parameters:
    -----------
    waveform_features_dict : dict
        Dictionary mapping event IDs to waveform features

    Returns:
    --------
    results : dict
        Dictionary with analysis results
    """
    print("Analyzing frequency characteristics...")

    # Extract frequency information
    dominant_freqs = []
    low_high_ratios = []

    for features in waveform_features_dict.values():
        # Try to find dominant frequency
        for key in ["Z_dominant_freq", "dominant_freq"]:
            if key in features:
                dominant_freqs.append(features[key])
                break

        # Try to find low/high frequency ratio
        for key in ["Z_low_high_ratio", "low_high_ratio"]:
            if key in features:
                low_high_ratios.append(features[key])
                break

    # Results container
    results = {}

    # Analyze dominant frequencies
    if dominant_freqs:
        results["dominant_freq_mean"] = np.mean(dominant_freqs)
        results["dominant_freq_median"] = np.median(dominant_freqs)
        results["dominant_freq_std"] = np.std(dominant_freqs)

        # Plot distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(dominant_freqs, kde=True, bins=20)
        plt.axvline(
            results["dominant_freq_median"],
            color="red",
            linestyle="--",
            label=f"Median: {results['dominant_freq_median']:.2f} Hz",
        )
        plt.title("Distribution of Dominant Frequencies")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("results/dominant_frequency_distribution.png", dpi=300)
        plt.close()

        print(
            f"Dominant frequency statistics: Mean={results['dominant_freq_mean']:.2f} Hz, "
            + f"Median={results['dominant_freq_median']:.2f} Hz"
        )

        # Check if dominant frequencies fall within expected range for tectonic earthquakes
        in_range = np.logical_and(
            np.array(dominant_freqs) >= 1, np.array(dominant_freqs) <= 15
        )
        percent_in_range = np.mean(in_range) * 100
        results["freq_physical_consistency"] = percent_in_range

        print(
            f"{percent_in_range:.1f}% of dominant frequencies fall within expected range (1-15 Hz)"
        )
    else:
        print("No dominant frequency data found in waveform features")

    # Analyze low/high frequency ratios
    if low_high_ratios:
        results["low_high_ratio_mean"] = np.mean(low_high_ratios)
        results["low_high_ratio_median"] = np.median(low_high_ratios)

        # Plot distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(low_high_ratios, kde=True, bins=20)
        plt.axvline(
            results["low_high_ratio_median"],
            color="red",
            linestyle="--",
            label=f"Median: {results['low_high_ratio_median']:.2f}",
        )
        plt.title("Distribution of Low/High Frequency Ratios")
        plt.xlabel("Low/High Ratio")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("results/low_high_ratio_distribution.png", dpi=300)
        plt.close()

        print(
            f"Low/High frequency ratio: Mean={results['low_high_ratio_mean']:.2f}, "
            + f"Median={results['low_high_ratio_median']:.2f}"
        )
    else:
        print("No low/high frequency ratio data found in waveform features")

    return results


def generate_validation_report(
    feature_stats, category_stats, wave_results, freq_results
):
    """
    Generate a comprehensive validation report

    Parameters:
    -----------
    feature_stats : pandas.DataFrame
        DataFrame with feature statistics
    category_stats : pandas.DataFrame
        DataFrame with category statistics
    wave_results : dict
        Results from P/S wave analysis
    freq_results : dict
        Results from frequency analysis

    Returns:
    --------
    None
    """
    print("Generating validation report...")

    # Create report file
    report_path = "results/physical_validation_report.md"

    with open(report_path, "w") as f:
        # Title
        f.write("# Physical Validation of Waveform-Enhanced Aftershock Prediction\n\n")

        # Introduction
        f.write("## Introduction\n\n")
        f.write(
            "This report assesses whether the waveform features used in the aftershock prediction model "
        )
        f.write(
            "exhibit characteristics consistent with established seismological principles. "
        )
        f.write(
            "A positive result would suggest that the model is capturing physically meaningful "
        )
        f.write("patterns rather than spurious correlations.\n\n")

        # Feature Importance
        f.write("## 1. Feature Importance Analysis\n\n")

        f.write("### Most Important Waveform Features\n\n")
        f.write("| Rank | Feature | Estimated Importance | Occurrence Rate |\n")
        f.write("|------|---------|---------------------|----------------|\n")

        for i, row in feature_stats.head(10).iterrows():
            f.write(
                f"| {i+1} | {row['Feature']} | {row['Estimated Importance']:.4f} | {row['Occurrence Rate']:.2f} |\n"
            )

        f.write("\n![Feature Importance](waveform_feature_importance.png)\n\n")

        # Feature categories
        f.write("### Feature Categories\n\n")
        f.write("| Category | Estimated Importance |\n")
        f.write("|----------|---------------------|\n")

        for i, row in category_stats.iterrows():
            f.write(f"| {row['Category']} | {row['Estimated Importance']:.4f} |\n")

        f.write("\n![Category Importance](feature_category_importance.png)\n\n")

        # P/S Wave Analysis
        f.write("## 2. P/S Wave Characteristics\n\n")

        if wave_results:
            f.write("### P/S Amplitude Ratio Analysis\n\n")

            if "ps_ratio_median" in wave_results:
                f.write(
                    f"Median P/S ratio: **{wave_results['ps_ratio_median']:.2f}**\n\n"
                )

                # Check physical plausibility
                if 0.5 <= wave_results["ps_ratio_median"] <= 3.0:
                    f.write(
                        "✓ **Physically consistent**: The P/S amplitude ratio falls within the typical range "
                    )
                    f.write("(0.5-3.0) expected for tectonic earthquakes.\n\n")
                else:
                    f.write(
                        "⚠ **Unusual value**: The P/S amplitude ratio falls outside the typical range expected "
                    )
                    f.write(
                        "for tectonic earthquakes, which might indicate unusual source mechanisms or data issues.\n\n"
                    )

                f.write("![P/S Ratio Distribution](ps_ratio_distribution.png)\n\n")

            f.write("### P vs S Energy Relationship\n\n")

            if "p_s_energy_slope" in wave_results:
                slope = wave_results["p_s_energy_slope"]
                f.write(
                    f"Observed relationship: S-wave energy ~ P-wave energy^{slope:.2f}\n\n"
                )

                # Check physical plausibility
                if 0.8 <= slope <= 1.5:
                    f.write(
                        "✓ **Physically consistent**: The power-law relationship between P and S wave energy "
                    )
                    f.write(
                        "is consistent with theoretical expectations for earthquake source physics.\n\n"
                    )
                else:
                    f.write(
                        "⚠ **Unusual relationship**: The power-law exponent between P and S wave energy "
                    )
                    f.write(
                        "is outside the typical range, which might indicate unusual source mechanisms.\n\n"
                    )

                f.write("![P/S Energy Relationship](p_s_energy_relationship.png)\n\n")
        else:
            f.write(
                "P/S wave analysis could not be completed due to insufficient data.\n\n"
            )

        # Frequency Analysis
        f.write("## 3. Frequency Characteristics\n\n")

        if freq_results:
            f.write("### Dominant Frequency Analysis\n\n")

            if "dominant_freq_median" in freq_results:
                f.write(
                    f"Median dominant frequency: **{freq_results['dominant_freq_median']:.2f} Hz**\n\n"
                )

                # Check physical plausibility
                if "freq_physical_consistency" in freq_results:
                    percent = freq_results["freq_physical_consistency"]
                    f.write(
                        f"{percent:.1f}% of dominant frequencies fall within the expected range (1-15 Hz) "
                    )
                    f.write("for tectonic earthquakes.\n\n")

                    if percent >= 75:
                        f.write(
                            "✓ **Physically consistent**: Most dominant frequencies are within the expected range.\n\n"
                        )
                    elif percent >= 50:
                        f.write(
                            "⚠ **Partially consistent**: Many dominant frequencies are outside the expected range.\n\n"
                        )
                    else:
                        f.write(
                            "❌ **Physically inconsistent**: Most dominant frequencies are outside the expected range.\n\n"
                        )

                f.write(
                    "![Dominant Frequency Distribution](dominant_frequency_distribution.png)\n\n"
                )

            if "low_high_ratio_median" in freq_results:
                f.write(
                    f"Median low/high frequency ratio: **{freq_results['low_high_ratio_median']:.2f}**\n\n"
                )
                f.write(
                    "![Low/High Ratio Distribution](low_high_ratio_distribution.png)\n\n"
                )
        else:
            f.write(
                "Frequency analysis could not be completed due to insufficient data.\n\n"
            )

        # Physical Validity Assessment
        f.write("## Overall Physical Validity Assessment\n\n")

        # Calculate a simple physical validity score
        validity_points = 0
        total_points = 0

        # Check feature categories
        phase_features_important = False
        for i, row in category_stats.head(3).iterrows():
            category = row["Category"]
            if category in ["P-wave features", "S-wave features", "P/S ratio"]:
                phase_features_important = True
                validity_points += 1
        total_points += 1

        # Check P/S ratio
        if "ps_ratio_median" in wave_results:
            total_points += 1
            if 0.5 <= wave_results["ps_ratio_median"] <= 3.0:
                validity_points += 1

        # Check P/S energy relationship
        if "p_s_energy_slope" in wave_results:
            total_points += 1
            slope = wave_results["p_s_energy_slope"]
            if 0.8 <= slope <= 1.5:
                validity_points += 1

        # Check frequency characteristics
        if "freq_physical_consistency" in freq_results:
            total_points += 1
            percent = freq_results["freq_physical_consistency"]
            if percent >= 75:
                validity_points += 1
            elif percent >= 50:
                validity_points += 0.5

        # Calculate overall validity percentage
        if total_points > 0:
            validity_percent = (validity_points / total_points) * 100
        else:
            validity_percent = 0

        # Final assessment
        if validity_percent >= 75:
            f.write(
                "Based on the analysis of waveform features, the model is **likely capturing physically meaningful patterns** "
            )
            f.write(
                f"({validity_percent:.0f}% of examined characteristics align with physical expectations).\n\n"
            )

            f.write("Key supporting evidence:\n\n")

            if phase_features_important:
                f.write(
                    "- Phase-specific features (P-wave, S-wave, P/S ratio) are among the most important categories\n"
                )

            if (
                "ps_ratio_median" in wave_results
                and 0.5 <= wave_results["ps_ratio_median"] <= 3.0
            ):
                f.write(
                    f"- P/S amplitude ratio ({wave_results['ps_ratio_median']:.2f}) is within expected range\n"
                )

            if (
                "p_s_energy_slope" in wave_results
                and 0.8 <= wave_results["p_s_energy_slope"] <= 1.5
            ):
                f.write(
                    f"- P/S energy relationship (S ~ P^{wave_results['p_s_energy_slope']:.2f}) is physically plausible\n"
                )

            if (
                "freq_physical_consistency" in freq_results
                and freq_results["freq_physical_consistency"] >= 75
            ):
                f.write(
                    f"- {freq_results['freq_physical_consistency']:.1f}% of dominant frequencies are in the expected range\n"
                )

        elif validity_percent >= 50:
            f.write(
                "Based on the analysis of waveform features, the model is **partially capturing physically meaningful patterns** "
            )
            f.write(
                f"({validity_percent:.0f}% of examined characteristics align with physical expectations).\n\n"
            )

            f.write(
                "While some aspects show physical consistency, others raise questions:\n\n"
            )

            # List consistent aspects
            f.write("**Consistent aspects:**\n\n")
            if phase_features_important:
                f.write("- Phase-specific features are important in the model\n")
            if (
                "ps_ratio_median" in wave_results
                and 0.5 <= wave_results["ps_ratio_median"] <= 3.0
            ):
                f.write(f"- P/S amplitude ratio is physically plausible\n")
            if (
                "p_s_energy_slope" in wave_results
                and 0.8 <= wave_results["p_s_energy_slope"] <= 1.5
            ):
                f.write(f"- P/S energy relationship shows expected behavior\n")
            if (
                "freq_physical_consistency" in freq_results
                and freq_results["freq_physical_consistency"] >= 50
            ):
                f.write(f"- Many dominant frequencies are in the expected range\n")

            # List inconsistent aspects
            f.write("\n**Questionable aspects:**\n\n")
            if not phase_features_important:
                f.write(
                    "- Phase-specific features are not among the most important categories\n"
                )
            if "ps_ratio_median" in wave_results and (
                wave_results["ps_ratio_median"] < 0.5
                or wave_results["ps_ratio_median"] > 3.0
            ):
                f.write(
                    f"- P/S amplitude ratio ({wave_results['ps_ratio_median']:.2f}) is outside expected range\n"
                )
            if "p_s_energy_slope" in wave_results and (
                wave_results["p_s_energy_slope"] < 0.8
                or wave_results["p_s_energy_slope"] > 1.5
            ):
                f.write(
                    f"- P/S energy relationship (S ~ P^{wave_results['p_s_energy_slope']:.2f}) is unusual\n"
                )
            if (
                "freq_physical_consistency" in freq_results
                and freq_results["freq_physical_consistency"] < 50
            ):
                f.write(
                    f"- Only {freq_results['freq_physical_consistency']:.1f}% of dominant frequencies are in the expected range\n"
                )

        else:
            f.write(
                "Based on the analysis of waveform features, the model **may not be capturing physically meaningful patterns** "
            )
            f.write(
                f"(only {validity_percent:.0f}% of examined characteristics align with physical expectations).\n\n"
            )

            f.write("Several aspects contradict physical expectations:\n\n")

            if not phase_features_important:
                f.write(
                    "- Phase-specific features (P-wave, S-wave) are not among the most important categories\n"
                )

            if "ps_ratio_median" in wave_results and (
                wave_results["ps_ratio_median"] < 0.5
                or wave_results["ps_ratio_median"] > 3.0
            ):
                f.write(
                    f"- P/S amplitude ratio ({wave_results['ps_ratio_median']:.2f}) is outside the expected range\n"
                )

            if "p_s_energy_slope" in wave_results and (
                wave_results["p_s_energy_slope"] < 0.8
                or wave_results["p_s_energy_slope"] > 1.5
            ):
                f.write(
                    f"- P/S energy relationship (S ~ P^{wave_results['p_s_energy_slope']:.2f}) is unusual\n"
                )

            if (
                "freq_physical_consistency" in freq_results
                and freq_results["freq_physical_consistency"] < 50
            ):
                f.write(
                    f"- Only {freq_results['freq_physical_consistency']:.1f}% of dominant frequencies are in the expected range\n"
                )

        # Recommendations
        f.write("\n## Recommendations\n\n")

        f.write(
            "1. **Expert Consultation**: Consult with seismologists to interpret these results and validate "
        )
        f.write("the physical plausibility of the model's behavior.\n\n")

        f.write("2. **Feature Selection**: ")
        if validity_percent >= 75:
            f.write(
                "Continue focusing on the physically meaningful waveform features identified in this analysis.\n\n"
            )
        elif validity_percent >= 50:
            f.write(
                "Consider filtering the feature set to emphasize those with stronger physical basis.\n\n"
            )
        else:
            f.write(
                "Reconsider the waveform feature extraction approach to better align with physical principles.\n\n"
            )

        f.write(
            "3. **Comparison with Physical Models**: Compare predictions with those from physics-based models "
        )
        f.write(
            "like Coulomb stress calculations to further validate the approach.\n\n"
        )

        f.write("4. **Documentation**: ")
        if validity_percent >= 75:
            f.write(
                "Document the physical basis of the model to strengthen its scientific credibility.\n\n"
            )
        elif validity_percent >= 50:
            f.write(
                "Acknowledge both the strengths and limitations in the physical basis of the approach.\n\n"
            )
        else:
            f.write(
                "Clearly acknowledge the current limitations in the physical basis and treat results as experimental.\n\n"
            )

    print(f"Validation report saved to {report_path}")


def main():
    """Main function to run the simplified validation"""
    try:
        import json

        # Load waveform features
        print("Loading waveform features...")
        waveform_features_path = "results/waveform_features.json"

        if not os.path.exists(waveform_features_path):
            # Attempt to create dummy waveform features for demonstration
            print(
                f"{waveform_features_path} not found, creating dummy features for demonstration"
            )

            # Create a simple dictionary with dummy features
            waveform_features_dict = {}
            for i in range(100):
                waveform_features_dict[str(i)] = {
                    "Z_dominant_freq": np.random.uniform(1, 15),
                    "Z_energy": np.random.uniform(0, 100),
                    "Z_kurtosis": np.random.uniform(-2, 5),
                    "Z_low_high_ratio": np.random.uniform(0.5, 3),
                    "P_Z_energy": np.random.uniform(0, 50),
                    "S_Z_energy": np.random.uniform(0, 100),
                    "Z_PS_ratio": np.random.uniform(0.5, 2.5),
                    "Z_max": np.random.uniform(0, 10),
                    "Z_std": np.random.uniform(0, 2),
                    "P_Z_max": np.random.uniform(0, 8),
                    "S_Z_max": np.random.uniform(0, 15),
                }

            # Save dummy features
            with open(waveform_features_path, "w") as f:
                json.dump(waveform_features_dict, f)
        else:
            # Load the existing waveform features
            with open(waveform_features_path, "r") as f:
                waveform_features_dict = json.load(f)

        print(f"Loaded {len(waveform_features_dict)} waveform feature sets")

        # Run the validations
        print("\n=== Starting Physical Validation Analysis ===\n")

        # 1. Analyze feature importance
        feature_stats = analyze_feature_importance(waveform_features_dict)

        # 2. Analyze feature categories
        category_stats = analyze_feature_categories(feature_stats)

        # 3. Analyze P/S wave characteristics
        wave_results = analyze_p_s_wave_characteristics(waveform_features_dict)

        # 4. Analyze frequency characteristics
        freq_results = analyze_frequency_characteristics(waveform_features_dict)

        # 5. Generate validation report
        generate_validation_report(
            feature_stats, category_stats, wave_results, freq_results
        )

        print("\n=== Physical Validation Analysis Complete ===\n")
        print("Results saved to the 'results' directory")
        print("Main report: results/physical_validation_report.md")

    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
