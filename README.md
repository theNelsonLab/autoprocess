# Crystallography Data Processing Suite

## Overview
This suite consists of three Python scripts designed for automated processing and analysis of MicroED (Micro-Electron Diffraction) data:
- **autoprocess.py**: Core script for automated MicroED data processing
- **batch_reprocess.py**: Batch reprocessing tool with specific space group and unit cell parameters
- **mrc2tif.py**: Utility for converting MRC files to TIF format

## Requirements

### Software Dependencies
1. **XDS Software Suite**: Required for crystallographic data processing
   - Must be accessible via the `xds` command
   - Available from [xds.mr.mpg.de](https://xds.mr.mpg.de/)

2. **Pointless**: Required for space group analysis (pointless)

## Script Details

### 1. autoprocess.py

#### Description
Primary script for automated MicroED data processing using the XDS suite. Handles image conversion, indexing, integration, and initial analysis.

#### Key Features
- Automated conversion of .ser/.mrc files to XDS-compatible formats
- Dynamic optimization of XDS processing parameters
- Automatic space group determination using CCP4's pointless
- Support for multiple microscope configurations
- Comprehensive error handling and logging

#### Supported Microscopes
- Arctica-CETA
- Arctica-EM-core
- Talos-Apollo

#### Usage
```bash
autoprocess [options]

Options:
  --microscope MICROSCOPE    Choose instrument (default: Arctica-CETA)
  --rotation-axis AXIS      Override rotation axis
  --frame-size SIZE        Override frame size
  --signal-pixel VALUE     Override signal pixel value
  --min-pixel VALUE       Override minimum pixel value
  --background-pixel VALUE Override background pixel value
  --pixel-size VALUE      Override pixel size value
  --beam-center-x VALUE   Override beam center X coordinate
  --beam-center-y VALUE   Override beam center Y coordinate
  --detector-distance VALUE Override detector distance
  --exposure VALUE        Override exposure time
  --rotation VALUE        Override rotation value
```

### 2. batch_reprocess

#### Description
Tool for batch reprocessing of previously processed data with specific space group and unit cell parameters.

#### Key Features
- Reprocess multiple datasets with consistent parameters
- Specify space group and unit cell parameters
- Custom processing parameter optimization
- Detailed processing logs and statistics

#### Usage
```bash
batch_reprocess [options]

Options:
  --microscope MICROSCOPE    Choose instrument (default: Arctica-CETA)
  --space-gr NUMBER        Space group number
  --a VALUE               Unit cell parameter a
  --b VALUE               Unit cell parameter b
  --c VALUE               Unit cell parameter c
  --alpha VALUE           Unit cell angle alpha
  --beta VALUE            Unit cell angle beta
  --gamma VALUE           Unit cell angle gamma
  --folder NAME           Subfolder name for reprocessed data
  --default-params        Use default processing parameters
  --signal-pixel VALUE    Signal pixel value
  --min-pixel VALUE      Minimum pixel value
  --background-pixel VALUE Background pixel value (max 5)
```

### 3. mrc2tif

#### Description
Utility script for converting MRC movie files to TIF format with detailed verification and logging.

#### Key Features
- Single and multi-frame MRC file support
- Optional pedestal value addition
- Detailed conversion verification
- Comprehensive logging of conversion statistics
- Raw data conversion option

#### Usage
```bash
mrc2tif [options]

Options:
  --folder PATH           Path to folder containing MRC files
  --ped VALUE            Pedestal value to add (default: 0)
  --tif-name NAME        Base name for output TIF files
  --recursive            Search for MRC files recursively
  --raw                  Convert data without modifications
```

## File Naming Convention
### For .ser Files
```
sample-name_distance_rotation_exposure_additional-notes.ser
```
Example: `sample-mov1_960_0.3_3_n60top10_g8sp10_cryo.ser`
- `distance`: Detector distance in mm
- `rotation`: Rotation speed in degrees/second
- `exposure`: Exposure time in seconds

## Directory Structure
```
working_directory/
├── sample_name/
│   ├── images/
│   │   └── (converted image files)
│   ├── auto_process/
│   │   └── (XDS processing files)
│   └── batch_reprocess/
│       └── (reprocessed data files)
├── autoprocess_logs/
│   └── (processing log files)
└── logs/
    └── (conversion log files)
```

## Error Handling
- All scripts include comprehensive error handling and logging
- Detailed logs are generated in the respective log directories
- Processing statistics and verification results are recorded
- Failed processes are clearly identified in the logs

## Contributing
Contributions are welcome! Please submit issues and pull requests to the project repository.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Original autoprocess.py by [Jessica Burch](https://github.com/jess-burch)
- Enhanced by [Dmitry Eremin](https://github.com/mit-eremin)
- Nelson Lab for ongoing support and development

## Version History
- v2.0.0: Added batch processing and MRC conversion capabilities
- v1.0.0: Initial autoprocess.py release
