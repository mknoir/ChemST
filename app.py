# Purpose: To filter ADME data from ChEMBL database

import streamlit as st

import array

import math

from pathlib import Path

import plotly.express as px

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import matplotlib.patches as mpatches

import numpy as np

import pandas as pd

from rdkit import Chem

from rdkit.Chem import Descriptors, Draw, PandasTools

 

from rdkit.Chem import AllChem

from chembl_webresource_client.new_client import new_client

from tqdm.auto import tqdm

import os

 

st.title("Filtering ADME Data App")

tab1, tab2, tab3 = st.tabs(["Tutorial", "Test", "Graph"])

 

HERE = Path(os.getcwd())

DATA = HERE / "data"

 

smiles = [

    "CCC1C(=O)N(CC(=O)N(C(C(=O)NC(C(=O)N(C(C(=O)NC(C(=O)NC(C(=O)N(C(C(=O)N(C(C(=O)N(C(C(=O)N(C(C(=O)N1)C(C(C)CC=CC)O)C)C(C)C)C)CC(C)C)C)CC(C)C)C)C)C)CC(C)C)C)C(C)C)CC(C)C)C)C",

    "CN1CCN(CC1)C2=C3C=CC=CC3=NC4=C(N2)C=C(C=C4)C",

    "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(CCCC2(C)C)C)C)C",

    "CCCCCC1=CC(=C(C(=C1)O)C2C=C(CCC2C(=C)C)C)O",

]

names = ["cyclosporine", "clozapine", "beta-carotene", "cannabidiol"]

 

molecules = pd.DataFrame({"name": names, "smiles": smiles})

PandasTools.AddMoleculeColumnToFrame(molecules, "smiles")

 

with tab1:

    st.subheader("Let's start with some test data:")

    st.dataframe(molecules)

 

    st.write(molecules.to_html(escape=False), unsafe_allow_html=True)

 

    st.write("Now let's process the molecular data:")

    #Calculate the mol properties

    molecules["molecular_weight"] = molecules["ROMol"].apply(Descriptors.ExactMolWt)

    molecules["n_hba"] = molecules["ROMol"].apply(Descriptors.NumHAcceptors)

    molecules["n_hbd"] = molecules["ROMol"].apply(Descriptors.NumHDonors)

    molecules["logp"] = molecules["ROMol"].apply(Descriptors.MolLogP)

    # Colors are used for plotting the molecules later

    molecules["color"] = ["red", "green", "blue", "cyan"]

    # NBVAL_CHECK_OUTPUT

    molecules[["name","molecular_weight", "n_hba", "n_hbd", "logp"]]

    #full_preview

    st.write(molecules.to_html(escape=False), unsafe_allow_html=True)

 

    st.write("Time to get the LIpinski's rule of 5 properties and plot them")

    #Plot mol properties as bar plots

    ro5_properties = {

        "molecular_weight": (500, "molecular weight (Da)"),

        "n_hba": (10, "# HBA"),

        "n_hbd": (5, "# HBD"),

        "logp": (5, "logP"),

    }

 

    # Start 1x4 plot frame

    fig, axes = plt.subplots(figsize=(10, 2.5), nrows=1, ncols=4)

    x = np.arange(1, len(molecules) + 1)

    colors = ["red", "green", "blue", "cyan"]

 

    # Create subplots

    for index, (key, (threshold, title)) in enumerate(ro5_properties.items()):

        axes[index].bar([1, 2, 3, 4], molecules[key], color=colors)

        axes[index].axhline(y=threshold, color="black", linestyle="dashed")

        axes[index].set_title(title)

        axes[index].set_xticks([])

 

    # Add legend

    legend_elements = [

        mpatches.Patch(color=row["color"], label=row["name"]) for index, row in molecules.iterrows()

    ]

    legend_elements.append(Line2D([0], [0], color="black", ls="dashed", label="Threshold"))

    fig.legend(handles=legend_elements, bbox_to_anchor=(1.2, 0.8))

 

    # Fit subplots and legend into figure

    plt.tight_layout()

    plt.show()

    st.subheader("Lipinski's Rule of Five:")

    st.pyplot(plt)

 

    #Investigate compliance with ro5

    def calculate_ro5_properties(smiles):

        """

        Test if input molecule (SMILES) fulfills Lipinski's rule of five.

 

        Parameters

        ----------

        smiles : str

            SMILES for a molecule.

 

        Returns

        -------

        pandas.Series

            Molecular weight, number of hydrogen bond acceptors/donor and logP value

            and Lipinski's rule of five compliance for input molecule.

        """

        # RDKit molecule from SMILES

        molecule = Chem.MolFromSmiles(smiles)

        # Calculate Ro5-relevant chemical properties

        molecular_weight = Descriptors.ExactMolWt(molecule)

        n_hba = Descriptors.NumHAcceptors(molecule)

        n_hbd = Descriptors.NumHDonors(molecule)

        logp = Descriptors.MolLogP(molecule)

        # Check if Ro5 conditions fulfilled

        conditions = [molecular_weight <= 500, n_hba <= 10, n_hbd <= 5, logp <= 5]

        ro5_fulfilled = sum(conditions) >= 3

        # Return True if no more than one out of four conditions is violated

        return pd.Series(

            [molecular_weight, n_hba, n_hbd, logp, ro5_fulfilled],

            index=["molecular_weight", "n_hba", "n_hbd", "logp", "ro5_fulfilled"],

        )

    # NBVAL_CHECK_OUTPUT

    for name, smiles in zip(molecules["name"], molecules["smiles"]):

        st.write(f"Ro5 fulfilled for {name}: {calculate_ro5_properties(smiles)['ro5_fulfilled']}")

 

with tab2:

    st.subheader("With this tutorial over, it's time to test the app on your own data and get the ADME properties:")

    st.write("you can use the ChemBl app to fetch data for your target and download that processed data as a csv file")

    st.write("Don't worry about the red error at the bottom, it will disappear once you input your data and press the button")

 

    # File uploader for user to upload a CSV file

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

 

    # Display a button to process the uploaded file

    if uploaded_file is not None:

        if st.button("Process File"):

            # Read the uploaded CSV file into a DataFrame

            molecules = pd.read_csv(uploaded_file, index_col=0)

            # This takes a couple of seconds

            ro5_properties = molecules["smiles"].apply(calculate_ro5_properties)

            ro5_properties.head()

 

            # Display the shape and the first few rows of the DataFrame

            st.write("DataFrame Shape:", molecules.shape)

            st.write("First Few Rows:")

            st.dataframe(molecules.head())

            st.write("Ro5 Properties:")

            st.write("A checkmark means true and an x means false")

            st.dataframe(ro5_properties.head())

    else:

        st.info("Please upload a CSV file.")

 

    #concatenate molecules and ro5_properties

    molecules = pd.concat([molecules, ro5_properties], axis=1)

    st.write("Concatenated data/Molecules with Ro5 Properties:")

    st.dataframe(molecules.head())

 

    st.subheader("Let's see how many of these molecules actually pass the Ro5 test:")

    # Note that the column "ro5_fulfilled" contains boolean values.

    # Thus, we can use the column values directly to subset data.

    # Note that ~ negates boolean values.

    molecules_ro5_fulfilled = molecules[molecules["ro5_fulfilled"]]

    molecules_ro5_violated = molecules[~molecules["ro5_fulfilled"]]

    # Define a formatting function

    def format_boolean(value):

        color = "green" if value else "red"

        return f"{color}"

 

    # Apply the formatting function to the entire DataFrame

    styled_molecules_ro5_fulfilled = molecules_ro5_fulfilled.applymap(format_boolean)

    st.divider()

    st.write(f"# compounds in unfiltered data set: {molecules.shape[0]}")

    st.divider()

    st.write(f"# compounds in filtered data set: {molecules_ro5_fulfilled.shape[0]}")

    st.divider()

    st.write(f"# compounds not compliant with the Ro5: {molecules_ro5_violated.shape[0]}")

    st.divider()

    st.write(f"Ro5 compliance: {molecules_ro5_fulfilled.shape[0] / molecules.shape[0]:.2%}")

    st.subheader("Feel free to download the filtered data as a csv file:")

    st.dataframe(molecules_ro5_fulfilled)

 

def calculate_mean_std(dataframe):

    # Generate descriptive statistics for property columns

    stats = dataframe.describe()

    # Transpose DataFrame (statistical measures = columns)

    stats = stats.T

    # Select mean and standard deviation

    stats = stats[["mean", "std"]]

    return stats

def _scale_by_thresholds(stats, thresholds, scaled_threshold):

    # Raise error if scaling keys and data_stats indicies are not matching

    for property_name in stats.index:

        if property_name not in thresholds.keys():

            raise KeyError(f"Add property '{property_name}' to scaling variable.")

    # Scale property data

    stats_scaled = stats.apply(lambda x: x / thresholds[x.name] * scaled_threshold, axis=1)

    return stats_scaled    

def _define_radial_axes_angles(n_axes):

    """Define angles (radians) for radial (x-)axes depending on the number of axes."""

    x_angles = [i / float(n_axes) * 2 * math.pi for i in range(n_axes)]

    x_angles += x_angles[:1]

    return x_angles

def plot_radar(

    y,

    thresholds,

    scaled_threshold,

    properties_labels,

    y_max=None,

    output_path=None,

):

    """

    Plot a radar chart based on the mean and standard deviation of a data set's properties.

 

    Parameters

    ----------

    y : pd.DataFrame

        Dataframe with "mean" and "std" (columns) for each physicochemical property (rows).

    thresholds : dict of str: int

        Thresholds defined for each property.

    scaled_threshold : int or float

        Scaled thresholds across all properties.

    properties_labels : list of str

        List of property names to be used as labels in the plot.

    y_max : None or int or float

        Set maximum y value. If None, let matplotlib decide.

    output_path : None or pathlib.Path

        If not None, save plot to file.

    """

 

    # Define radial x-axes angles -- uses our helper function!

    x = _define_radial_axes_angles(len(y))

    # Scale y-axis values with respect to a defined threshold -- uses our helper function!

    y = _scale_by_thresholds(y, thresholds, scaled_threshold)

    # Since our chart will be circular we append the first value of each property to the end

    y = pd.concat([y, y.head(1)])

 

    # Set figure and subplot axis

    plt.figure(figsize=(6, 6))

    ax = plt.subplot(111, polar=True)

 

    # Plot data

    ax.fill(x, [scaled_threshold] * len(x), "cornflowerblue", alpha=0.2)

    ax.plot(x, y["mean"], "b", lw=3, ls="-")

    ax.plot(x, y["mean"] + y["std"], "orange", lw=2, ls="--")

    ax.plot(x, y["mean"] - y["std"], "orange", lw=2, ls="-.")

 

    #plot cosmetics

    # Set 0° to 12 o'clock

    ax.set_theta_offset(math.pi / 2)

    # Set clockwise rotation

    ax.set_theta_direction(-1)

 

    # Set y-labels next to 180° radius axis

    ax.set_rlabel_position(180)

    # Set number of radial axes' ticks and remove labels

    plt.xticks(x, [])

    # Get maximal y-ticks value

    if not y_max:

        y_max = int(ax.get_yticks()[-1])

    # Set axes limits

    plt.ylim(0, y_max)

    # Set number and labels of y axis ticks

    plt.yticks(

        range(1, y_max),

        ["5" if i == scaled_threshold else "" for i in range(1, y_max)],

        fontsize=16,

    )

 

    # Draw ytick labels to make sure they fit properly

    # Note that we use [:1] to exclude the last element which equals the first element (not needed here)

    for i, (angle, label) in enumerate(zip(x[:-1], properties_labels)):

        if angle == 0:

            ha = "center"

        elif 0 < angle < math.pi:

            ha = "left"

        elif angle == math.pi:

            ha = "center"

        else:

            ha = "right"

        ax.text(

            x=angle,

            y=y_max + 1,

            s=label,

            size=16,

            horizontalalignment=ha,

            verticalalignment="center",

        )

 

    # Add legend relative to top-left plot

    labels = ("mean", "mean + std", "mean - std", "rule of five area")

    ax.legend(labels, loc=(1.1, 0.7), labelspacing=0.3, fontsize=16)

 

    # Save plot - use bbox_inches to include text boxes

    if output_path:

        plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)

 

    st.pyplot(plt.gcf()) # instead of plt.show()

thresholds = {"molecular_weight": 500, "n_hba": 10, "n_hbd": 5, "logp": 5}

scaled_threshold = 5

properties_labels = [

    "Molecular weight (Da) / 100",

    "# HBA / 2",

    "# HBD",

    "LogP",

]

y_max = 7

y_n_max = 15

 

#Time to calculate Statistics and graph the data

with tab3:

    st.subheader("Now that we have the filtered data, let's calculate some statistics and graph the data")

    st.write("Here is a small key for the statistics:")

    st.write("Molecular weight =MWT")

    st.write("number of Hydrogen bond acceptors = HBA")

    st.write("number of Hydrogen bond donors = HBD")

    st.write("octanol-water coefficient = LogP")

    molecules_ro5_fulfilled_stats = calculate_mean_std(

    molecules_ro5_fulfilled[["molecular_weight", "n_hba", "n_hbd", "logp"]]

)

    st.write("Ro5 Fulfilled Statistics:")

    row_names = {

        'n_hba': '# Hydrogen bond acceptors',

        "n_hbd": "# Hydrogen bond donors",

    }

    molecules_ro5_fulfilled = molecules_ro5_fulfilled.rename(index=row_names)

    st.dataframe(molecules_ro5_fulfilled_stats)

    st.subheader("plotfor molecules that fullfill the Ro5:")

 

    plot_radar(

        molecules_ro5_fulfilled_stats,

        thresholds,

        scaled_threshold,

        properties_labels,

        y_max,

    )

   

 

    molecules_ro5_violated_stats = calculate_mean_std(

    molecules_ro5_violated[["molecular_weight", "n_hba", "n_hbd", "logp"]]

)

    st.write("Ro5 Violated Statistics:")

    molecules_ro5_violated=molecules_ro5_violated.rename(index=row_names)

 

    st.dataframe(molecules_ro5_violated_stats)

   

    st.subheader("plot for molecules that violate the Ro5:")

    plot_radar(

        molecules_ro5_violated_stats,

        thresholds,

        scaled_threshold,

        properties_labels,

        y_n_max,

    )

 