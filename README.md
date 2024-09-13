Efflux pumps

Matthew Gerry

Python scripts for carrying out calculations associated with the efflux pump specificity and tunablity project. Only common libraries such as numpy and matplotlib are used, and the project is run simply by running the python scripts in the command line. Store all associated files in a common directory.

The files parameters.py, rate_matrix.py, and efflux_pumps.py contain useful functions needed for all the desired calculations.

Running the scripts plot_efflux_3.py, plot_efflux_5.py, plot_efflux_7.py, and kd_at_jmax.py as is will generate the plots present in the paper (https://www.biorxiv.org/content/10.1101/2024.06.11.598359v1), including those the supporting information file. Additionally, the values of parameters may be modified within the scripts to explore further. Some additional functions are included, e.g., for examining the widths of the efflux vs. KD curves as a measure of specificity, etc.

I have written and managed all the code, but the project was a collaboration with Duncan Kirby, Boian S. Alexandrov, Dvira Segal, and Anton Zilman.