import streamlit as st
import array
import math
from pathlib import Path
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
    # Calculate the mol properties
    molecules["molecular_weight"] = molecules["ROMol"].apply(Descriptors.ExactMolWt)
    molecules["n_hba"] = molecules["ROMol"].apply(Descriptors.NumHAcceptors)
    molecules["n_hbd"] = molecules["ROMol"].apply(Descriptors.NumHDonors)
    molecules["logp"] = molecules["ROMol"].apply(Descriptors.MolLogP)
    # Colors are used for plotting the molecules later
    molecules["color"] = ["red", "green", "blue", "cyan"]
    molecules[["name", "molecular_weight", "n_hba", "n_hbd", "logp"]]
    st.write(molecules.to_html(escape=False), unsafe_allow_html=True)

    st.write("Time to get the Lipinski's rule of 5 properties and plot them")
    # Plot mol properties as bar plots
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

    # Investigate compliance with ro5
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

    for name, smiles in zip(molecules["name"], molecules["smiles"]):
        st.write(f"Ro5 fulfilled for {name}: {calculate_ro5_properties(smiles)['ro5_fulfilled']}")

with tab2:
    st.subheader("With this tutorial over, it's time to test the app on your own data and get the ADME properties:")
    st.write("you can use the ChemBl app to fetch data for your target and download that processed data as a csv file")
    st.markdown("[Here is a link to the ChemBL app](https://chembl.streamlit.app/)")
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
            st.write("I don't see the structures so let's fetch them real quick")
            moleculesH = molecules.head()
            PandasTools.AddMoleculeColumnToFrame(moleculesH, "smiles")
            st.write(moleculesH.to_html(escape=False), unsafe_allow_html=True)
            st.write("Ro5 Properties:")
            st.write("A checkmark means true and an x means false")
            st.dataframe(ro5_properties.head())
    else:
        st.info("Please upload a CSV file.")

    # Concatenate molecules and ro5_properties
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