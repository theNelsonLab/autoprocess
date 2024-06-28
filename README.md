# autoprocess.py

## Description
Autoprocess is a Python script designed to automate the initial steps of MicroED data processing utilizing XDS. The script takes input data files as .ser movies in a specific format, converts them to images, and attempts to index and integrate by XDS independently.

It is designed to be used as a tool that aids in quickly deciding whether data collection should be continued or not, or when learning how to process microED data.


## Requirements
1. Properly installed XDS software (can be obtained from: https://xds.mr.mpg.de/) that can be called from Linux or Unix using the "xds" command
2. The ser2smv image conversion program (can be obtained from: https://cryoem.ucla.edu/downloads/snapshots)
3. Python 3

## Usage
1. Name .ser files according to the following format: your-name-here_detectordistance_rotationspeed_integrationtime_your-notes-here.ser. Example: Example-mov1_960_0.3_3_n60deg-to-p10deg_gun8spot10_cryo.ser
Rotation speed is degrees/second, exposure time is in seconds, and detector distance is in mm.
    
2. Navigate to the directory containing your ser files and copy the autoprocess.py script. Edits to the script may need to be performed to locate ser2smv or for differences in microscope configurations.

3. Run the autoprocess.py script:
   ```shell
   python autoprocess.py
   ```

## Contributing
Contributions to autoprocess.py are welcome! If you encounter any issues or have suggestions for improvements, please open an issue on the [GitHub repository](https://github.com/jess-burch/microed/issues).

## License
This project is licensed under the [MIT License](https://github.com/jess-burch/microed/blob/main/LICENSE).

## Acknowledgments
autoprocess.py was developed by [Jessica Burch](https://github.com/jess-burch).

Please note that this script is provided as-is, without any warranties. Use it at your own risk.