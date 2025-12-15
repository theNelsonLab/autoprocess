# pyautoprocess

A Python package for automated processing and analysis of MicroED (Micro-Electron Diffraction) data.

## Overview
This package provides a comprehensive suite of tools for automated MicroED data processing:
- **autoprocess**: Core script for automated MicroED data processing
- **image_process**: Preconverted image processing and quality analysis tool
- **batch_reprocess**: Batch reprocessing tool with specific space group and unit cell parameters
- **mrc2tif**: Utility for converting MRC files to TIF format
- **ser2tif**: Utility for converting SER files to TIF format
- **tvips2tif**: Utility for converting TVIPS files to TIF format

## Installation

Install from PyPI:
```bash
pip install pyautoprocess
```

Or install from source:
```bash
git clone https://github.com/theNelsonLab/autoprocess.git
cd autoprocess
pip install .

## Requirements

### Python Dependencies
- Python >= 3.8
- numpy ~= 2.0
- mrcfile ~= 1.5
- tifffile >= 2025.1.10
- seremi == 0.0.1
- scipy >= 1.7.0
- scikit-image >= 0.19.0

### External Software Dependencies
1. **XDS Software Suite**: Required for crystallographic data processing
   - Must be accessible via the `xds` command
   - Available from [xds.mr.mpg.de](https://xds.mr.mpg.de/)

2. **Pointless**: Required for space group analysis (pointless)
   - Part of the CCP4 software suite

## Script Details

### 1. autoprocess

#### Description
Primary tool for automated MicroED data processing using the XDS suite. Handles image conversion, indexing, integration, space group analysis, and scaling with intelligent parameter optimization.

#### Key Features
- **File Processing**: Processes .mrc/.ser/.tvips files from specified paths or current directory
- **Automatic Image Conversion**: Converts source files to XDS-compatible TIF format with validation
- **Diffraction Quality Analysis**: Optional frame quality assessment and intelligent frame selection (--dqa)
- **Dynamic XDS Optimization**: Automatic parameter adjustment for indexing, integration, and scaling
- **Advanced Space Group Analysis**: Intelligent space group optimization using lattice analysis
- **Pointless Integration**: Optional CCP4 pointless analysis for space group validation (--pointless)
- **Parallel Processing**: Support for parallel XDS execution (--parallel)
- **Comprehensive Logging**: Detailed processing logs and progress tracking
- **Microscope Configurations**: Built-in support for multiple instrument configurations
- **Reprocessing Control**: Skip already processed files or force reprocessing (--reprocess)

#### Usage
```bash
autoprocess [paths] [options]

Positional Arguments:
  paths                    Path(s) to process: single .mrc/.ser/.tvips file, folder containing files,
                          or multiple files/folders. If not specified, processes all files
                          in current directory.

Microscope Configuration:
  --microscope-config CONFIG  Choose instrument configuration (default: default)
  --config-file FILE          Path to microscope configuration file

Processing Control:
  --reprocess                 Reprocess files even if they have been processed before
  --pointless                 Run pointless for space group analysis
  --parallel                  Use parallel XDS (xds_par) instead of serial XDS
  --dqa                       Enable diffraction quality analysis and frame selection
  --verbose                   Enable verbose logging for detailed conversion validation

XDS Parameters Override:
  --rotation-axis AXIS        Override rotation axis
  --frame-size SIZE          Override frame size
  --signal-pixel VALUE       Override signal pixel value (XDS parameter)
  --min-pixel VALUE         Override minimum pixel value (XDS parameter)
  --background-pixel VALUE   Override background pixel value (XDS parameter)
  --pixel-size VALUE        Override pixel size value
  --wavelength VALUE        Override wavelength value
  --beam-center-x VALUE     Override beam center X coordinate
  --beam-center-y VALUE     Override beam center Y coordinate
  --file-extension EXT      Override input file extension

Experimental Parameters Override:
  --detector-distance VALUE  Override detector distance (in mm)
  --exposure VALUE           Override exposure time
  --rotation VALUE          Override rotation value

Resolution Control:
  --res-range VALUE         Manual resolution range in Angstroms (overrides calculated values)
  --min-res VALUE          Minimum resolution for XSCALE in Angstroms (independent from XDS)

Advanced XDS Parameters:
  --friedel BOOL           Set Friedel's law for XDS (true or false, default: true)
```


### 2. image_process

#### Description
Processing tool for pre-converted crystallography images with backup management, quality analysis, and flexible format support.

#### Key Features
- **Pre-Converted Processing**: Process existing TIF/IMG image files without conversion
- **Format Flexibility**: Support for both TIF and SMV (.img) formats (--smv)
- **Quality Analysis Integration**: Optional diffraction quality analysis with frame selection (--dqa)
- **Frame Trimming**: Precise frame range control with --trim-front and --trim-end
- **Backup Management**: Automatic backup system with organized archive folders
- **Multi-Path Processing**: Process multiple directories or specific paths simultaneously
- **Microscope Configurations**: Support for all standard microscope configurations
- **Processing Isolation**: Separate output folders (auto_process_direct) to avoid conflicts

#### Usage
```bash
image_process [paths] [options]

Positional Arguments:
  paths                    Path(s) to process: folders containing pre-converted images.
                          If not specified, processes all suitable folders in current directory.

File Format Options:
  --smv                    Process SMV (.img) files instead of TIF files

Frame Selection:
  --trim-front N           Number of frames to trim from the start of the range (default: 0)
  --trim-end N             Number of frames to trim from the end of the range (default: 0)
  --dqa                    Enable diffraction quality analysis and frame selection

Microscope Configuration:
  --microscope-config CONFIG  Choose instrument configuration (default: default)
  --config-file FILE          Path to microscope configuration file

Processing Control:
  --pointless                 Run pointless for space group analysis
  --parallel                  Use parallel XDS (xds_par) instead of serial XDS
  --verbose                   Enable verbose logging

XDS Parameters Override:
  --rotation-axis AXIS        Override rotation axis
  --frame-size SIZE          Override frame size
  --signal-pixel VALUE       Override signal pixel value
  --min-pixel VALUE         Override minimum pixel value
  --background-pixel VALUE   Override background pixel value
  --pixel-size VALUE        Override pixel size value
  --wavelength VALUE        Override wavelength value
  --beam-center-x VALUE     Override beam center X coordinate
  --beam-center-y VALUE     Override beam center Y coordinate

Experimental Parameters Override:
  --detector-distance VALUE  Override detector distance (in mm)
  --exposure VALUE           Override exposure time
  --rotation VALUE          Override rotation value

Resolution Control:
  --res-range VALUE         Manual resolution range in Angstroms (overrides calculated values)
  --min-res VALUE          Minimum resolution for XSCALE in Angstroms (independent from XDS)

Advanced XDS Parameters:
  --friedel BOOL           Set Friedel's law for XDS (true or false, default: true)

Examples:
  # Process TIF images in current directory
  image_process

  # Process SMV files with frame trimming
  image_process --smv --trim-front 5 --trim-end 10 /path/to/images

  # Quality analysis with multiple paths
  image_process --dqa /data/sample1 /data/sample2 /data/sample3

  # Advanced processing with parameter overrides
  image_process --parallel --pointless --signal-pixel 6 --trim-front 3 /path/to/data
```

### 3. batch_reprocess

#### Description
Advanced tool for batch reprocessing of crystallography data with specific space group and unit cell parameters. Supports both manual parameter specification and smart automatic detection mode.

#### Key Features
- **Smart Detection Mode**: Automatically detect optimal parameters from existing processing results (--smart)
- **Manual Parameter Mode**: Specify exact space group and unit cell parameters (--manual, default)
- **Interactive Input**: Prompts for missing parameters when not provided via command line
- **Flexible Reindexing**: Support for custom reindexing matrices
- **Batch Processing**: Process multiple datasets with consistent or adaptive parameters
- **Archive Management**: Organized backup and folder management for reprocessed data

#### Usage
```bash
batch_reprocess [paths] [options]

Positional Arguments:
  paths                    Path(s) to reprocess: folders containing processed data

Processing Mode:
  --manual                 Manual parameter specification mode (default)
  --smart                  Smart detection mode - automatically detect parameters

Crystallographic Parameters (Manual Mode):
  --space-gr NUMBER        Space group number
  --a VALUE               Unit cell parameter a (Å)
  --b VALUE               Unit cell parameter b (Å)
  --c VALUE               Unit cell parameter c (Å)
  --alpha VALUE           Unit cell angle alpha (degrees)
  --beta VALUE            Unit cell angle beta (degrees)
  --gamma VALUE           Unit cell angle gamma (degrees)
  --folder NAME           Subfolder name for reprocessed data
  --reidx "MATRIX"        Reindexing matrix as space-separated string
                          (e.g., "1 0 0 0 0 1 0 0 0 0 1 0")

Examples:
  # Smart mode - auto-detect parameters
  batch_reprocess --smart /path/to/data/folders

  # Manual mode with specific parameters
  batch_reprocess --space-gr 19 --a 50.1 --b 60.2 --c 70.3 \
                  --alpha 90 --beta 90 --gamma 90 --folder reprocess_P212121

  # Interactive mode (prompts for missing parameters)
  batch_reprocess /path/to/data
```

### 4. mrc2tif

#### Description
Precision utility for converting MRC movie files to TIF format with comprehensive verification, statistics, and data integrity checking.

#### Key Features
- **Multi-Format Support**: Single and multi-frame MRC file processing
- **Data Type Preservation**: Intelligent data type handling and conversion
- **Pedestal Addition**: Optional pedestal value addition with range validation
- **Raw Conversion Mode**: Preserve original data without type conversion (--raw)
- **Comprehensive Verification**: Frame-by-frame conversion validation with statistical analysis
- **Detailed Logging**: Split logging (file and console) with conversion statistics
- **Recursive Processing**: Process files in subdirectories (--recursive)
- **Custom Naming**: Flexible output file naming with --tif-name option
- **Data Range Validation**: Automatic checking for uint16 overflow/underflow
- **Directory Management**: Automatic 'images' subdirectory creation

#### Usage
```bash
mrc2tif [options]

Options:
  --folder PATH           Path to folder containing MRC files (default: current directory)
  --ped VALUE            Pedestal value to add to each pixel (default: 0)
  --tif-name NAME        Base name for output TIF files (default: same as MRC filename)
  --recursive            Search for MRC files recursively in subdirectories
  --raw                  Convert data without any type conversion or modifications

Examples:
  # Convert all MRC files in current directory
  mrc2tif

  # Convert with pedestal addition
  mrc2tif --ped 100 --folder /path/to/mrc/files

  # Recursive conversion with custom naming
  mrc2tif --recursive --tif-name sample_data --folder /data/root

  # Raw conversion preserving original data types
  mrc2tif --raw --folder /path/to/data
```

### 5. ser2tif

#### Description
Specialized utility for converting SER (Serial Electron Microscopy) movie files to TIF format using the seremi library for precise SER file handling and frame extraction.

#### Key Features
- **SER Format Expertise**: Native SER file format support using seremi library
- **TIA Compatibility**: Full compatibility with TIA and other SER-generating software
- **Frame Extraction**: Automatic individual frame extraction and processing
- **Data Integrity**: Comprehensive conversion verification and validation
- **Pedestal Support**: Optional pedestal value addition with validation
- **Raw Conversion**: Raw data conversion mode preserving original formats (--raw)
- **Detailed Logging**: Split logging system with comprehensive statistics
- **Recursive Processing**: Search subdirectories for SER files (--recursive)
- **Custom Naming**: Flexible output file naming control
- **Directory Management**: Automatic 'images' subdirectory organization

#### Usage
```bash
ser2tif [options]

Options:
  --folder PATH           Path to folder containing SER files (default: current directory)
  --ped VALUE            Pedestal value to add to each pixel (default: 0)
  --tif-name NAME        Base name for output TIF files (default: same as SER filename)
  --recursive            Search for SER files recursively in subdirectories
  --raw                  Convert data without any type conversion or modifications

Examples:
  # Convert all SER files in current directory
  ser2tif

  # Convert with pedestal and custom naming
  ser2tif --ped 50 --tif-name converted_frames --folder /data/ser_files

  # Recursive search with raw conversion
  ser2tif --recursive --raw --folder /microscopy/data
```

### 6. tvips2tif

#### Description
Utility for converting TVIPS movie files to TIF format with comprehensive verification and data integrity checking.

#### Key Features
- **TVIPS Format Support**: Native TVIPS file format support
- **Frame Extraction**: Automatic individual frame extraction and processing
- **Data Integrity**: Comprehensive conversion verification and validation
- **Pedestal Support**: Optional pedestal value addition with validation
- **Raw Conversion**: Raw data conversion mode preserving original formats (--raw)
- **Detailed Logging**: Split logging system with comprehensive statistics
- **Recursive Processing**: Search subdirectories for TVIPS files (--recursive)
- **Custom Naming**: Flexible output file naming control
- **Directory Management**: Automatic 'images' subdirectory organization

#### Usage
```bash
tvips2tif [options]

Options:
  --folder PATH           Path to folder containing TVIPS files (default: current directory)
  --ped VALUE            Pedestal value to add to each pixel (default: 0)
  --tif-name NAME        Base name for output TIF files (default: same as TVIPS filename)
  --recursive            Search for TVIPS files recursively in subdirectories
  --raw                  Convert data without any type conversion or modifications

Examples:
  # Convert all TVIPS files in current directory
  tvips2tif

  # Convert with pedestal and custom naming
  tvips2tif --ped 100 --tif-name converted_frames --folder /data/tvips_files

  # Recursive search with raw conversion
  tvips2tif --recursive --raw --folder /microscopy/data
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
This project is licensed under the GPL-3.0-or-later License.

## Authors and Contributors
- **Sam Foxman** (sfoxman@caltech.edu) - Package maintainer and development lead
- **Dmitry Eremin** (eremin@caltech.edu) - Core development and enhancements
- **Jessica Burch** - Original autoprocess.py implementation

## Acknowledgments
- Nelson Lab at Caltech for ongoing support and development
- XDS developers for data processing capabilities
- CCP4 Software Suite for crystallographic tools

## Version History
- **v0.3.2**: Resolution control and TVIPS support
  - Added dynamic XSCALE resolution shell commenting with --min-res argument
  - Enabled TVIPS file format support throughout processing pipeline
  - Added F30-TVIPS-SM microscope configuration
  - Fixed empty auto_process folder backup failures
  - Unified auto_process folder structure with backward compatibility migration
  - Added --friedel parameter for Friedel's law control
- **v0.2.0**: Major refactoring and DQA implementation
  - Implemented quality analysis and diffraction assessment
  - Added modular architecture with separate core, config, and UI modules
- **v0.1.x**: Enhanced autoprocess functionality
  - Restructured as installable Python package (pyautoprocess)
  - Added image_process tool for pre-converted image processing
  - Enhanced error handling and logging capabilities
  - Added support for frame trimming and SMV format
  - Added pointless and parallel processing flags
  - Improved parameter handling and microscope configurations
- **v0.0.x**: Initial development releases
  - Core autoprocess.py functionality
  - Basic batch reprocessing capabilities
  - MRC to TIF conversion utilities
