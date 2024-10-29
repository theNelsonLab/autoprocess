# autoprocess.py

## Description
**autoprocess.py** is a Python script designed for the automation of initial MicroED data processing using the XDS suite. It streamlines steps like image conversion, indexing, and integration, offering fast feedback on data quality. It is particularly useful for determining if data collection should continue or as a learning tool for processing MicroED data.

## Key Features
- **Automated .ser to .img Conversion**: Converts .ser movie files to .img format compatible with XDS using `ser2smv`.
- **Dynamic XDS Optimization**: Adjusts indexing parameters iteratively for improved data processing outcomes.
- **Automatic Scaling and Conversion**: Scales processed data in XSCALE and converts it for further analysis in SHELX and CCP4's `pointless`.

## Requirements
1. **XDS Software**: Properly installed XDS (available from [xds.mr.mpg.de](https://xds.mr.mpg.de/)) accessible via the `xds` command in Linux/Unix.
2. **ser2smv Conversion Program**: Required to convert .ser files to .img format. Available for download from [UCLA CryoEM](https://cryoem.ucla.edu/downloads/snapshots).
3. **Python 3**: Ensure Python 3 is installed with necessary permissions.

## Usage
### Preparing .ser Files
1. **File Naming Convention**: Name .ser files using the following format:
   ```
   sample-name_distance_rotation_exposure_additional-notes.ser
   ```
   - Example: `sample-mov1_960_0.3_3_n60top10_g8sp10_cryo.ser`
   - Where:
     - `distance`: Detector distance in mm.
     - `rotation`: Rotation speed in degrees/second.
     - `exposure`: Exposure time in seconds.

### Running autoprocess.py
1. **Set Up**: Place `autoprocess.py` in the directory containing your .ser files. You may need to edit the script to specify paths to `ser2smv` or adjust settings for your microscope.
2. **Execute**: Run the script as follows:
   ```shell
   python autoprocess.py
   ```
   The script will process all .ser files in the directory, automatically converting, indexing, and preparing files for further data analysis.

## Workflow Overview
- **Conversion**: Converts each .ser movie into .img files compatible with XDS.
- **Indexing & Integration**: Initiates XDS with default settings, iteratively adjusting if needed to improve indexing success.
- **Space Group Determination**: Uses CCP4's `pointless` to analyze symmetry and identify possible space groups.
- **Output Files**: Saves processed data in appropriate directories and generates `.hkl` files for SHELX processing.

## Contributing
Contributions to `autoprocess.py` are welcome! For feature requests, bug reports, or improvements, please open an issue on the [GitHub repository](https://github.com/theNelsonLab/autoprocess/issues).

## License
This project is licensed under the [MIT License](https://github.com/theNelsonLab/autoprocess/blob/main/LICENSE).

## Acknowledgments
`autoprocess.py` was originally developed by [Jessica Burch](https://github.com/jess-burch) and enhanced by [Dmitry Eremin](https://github.com/mit-eremin) . Special thanks to the Nelson Lab for supporting ongoing improvements.

---

This README now includes more structured information about features, usage, and key steps in the script's processing workflow. Let me know if you’d like further customization!