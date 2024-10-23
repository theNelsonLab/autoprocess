"""
autoprocess_version3 modified by Dmitry Eremin from original file by Jessica Burch
"""
import os
import random
from subprocess import run

def convert(sample_movie, distance, rotation, exposure, pix_size, i):
    #Following assumes you followed the naming convention of "name_distance_rotation_exposure.ser"
    # P = pixel size, B = binning, r = rotation rate, w = wavelength in A,
    # d = detector distance in mm, E = exposure time, M = offset
    ser2smv = "/groups/NelsonLab/programs/ser2smv"
    e_wavelength = str(0.0251)
    pix_size = str(pix_size)
    run([ser2smv, "-P", pix_size, "-B", "2", "-r", rotation, "-w", e_wavelength, 
                     "-d", distance, "-E", exposure, "-M", "200", "-v", 
                     "-o", sample_movie + "_###.img", os.path.join("..", i)], stdout=open(os.devnull, 'wb'))

def process_movie():
    current_path = os.getcwd()
    file = os.listdir()

    s_pix = 7
    min_pix = 7
    bkgrnd_pix = 4
    pix_size = 0.028


    for i in file:
        if i.endswith(".ser"):
            split = i.split("_")
            if len(split) < 4:
                print(f"Skipping {i}: unexpected filename format.")
                continue
            sample_movie, distance, rotation, exposure = split[0], split[1], split[2], split[3]
            resolution_range = float(distance) * 0.0009 - 0.1
            test_resolution_range = round(float(distance) * 0.0009 - 0.1, 2)
            if resolution_range < 0 or test_resolution_range < 0:
                print(f"Skipping {i}: resolution range is negative.")
                continue
            if not os.path.exists(sample_movie):
                os.makedirs(sample_movie, exist_ok=True)
                movie_path = os.path.join(current_path, sample_movie)
                os.rename(i, os.path.join(movie_path, i))

                image_path = os.path.join(movie_path, "images")
                os.makedirs(image_path, exist_ok=True)
                os.chdir(image_path)

                convert(sample_movie, distance, rotation, exposure, pix_size, i)
                image_number = str(len(os.listdir(image_path)))

                auto_process_path = os.path.join(movie_path, "auto_process")
                os.makedirs(auto_process_path, exist_ok=True)
                os.chdir(auto_process_path)
                with open("XDS.INP", "w") as xds_inp:
                    data_path = os.path.join(image_path, sample_movie)
                    xds_inp.write(f"""JOB= XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT
!JOB=DEFPIX INTEGRATE CORRECT
ORGX= 1018 ORGY= 1008 ! check X and Y of beam center
DETECTOR_DISTANCE= {float(distance)}
OSCILLATION_RANGE= {float(exposure) * float(rotation)}
X-RAY_WAVELENGTH= 0.0251000002

NAME_TEMPLATE_OF_DATA_FRAMES= {data_path}_???.img
BACKGROUND_RANGE=1 10
!DELPHI=15
!SPACE_GROUP_NUMBER=0
!UNIT_CELL_CONSTANTS= 1 1 1 90 90 90
INCLUDE_RESOLUTION_RANGE= 40 {resolution_range}
TEST_RESOLUTION_RANGE= 40 {test_resolution_range}
TRUSTED_REGION=0.0 1.2
VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS=6000. 30000.  ! parameters for detector and beamline
DETECTOR= ADSC MINIMUM_VALID_PIXEL_VALUE= 1 OVERLOAD= 65000
SENSOR_THICKNESS= 0.01
NX= 2048 NY= 2048 QX= {pix_size} QY= {pix_size}  ! detector parameters
ROTATION_AXIS=0 -1 0  ! this is a common reason data doesn't index, check for each scope
DIRECTION_OF_DETECTOR_X-AXIS=1 0 0
DIRECTION_OF_DETECTOR_Y-AXIS=0 1 0
INCIDENT_BEAM_DIRECTION=0 0 1
FRACTION_OF_POLARIZATION=0.98
POLARIZATION_PLANE_NORMAL=0 1 0
REFINE(IDXREF)=CELL BEAM ORIENTATION AXIS  ! DISTANCE
REFINE(INTEGRATE)= DISTANCE BEAM ORIENTATION  ! AXIS CELL
REFINE(CORRECT)=CELL BEAM ORIENTATION AXIS  ! DISTANCE

DATA_RANGE= 1 {image_number}
SPOT_RANGE= 1 {image_number}
BACKGROUND_PIXEL={bkgrnd_pix}
SIGNAL_PIXEL= {s_pix}
MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= {min_pix}
!
!""")

                    xds_inp.close()
                xds_out_path = os.path.join(auto_process_path, "XDS.LP")
                with open(xds_out_path, "w+") as xds_out:
                    print(f"Processing {sample_movie}...")
                    run("xds", stdout=xds_out)
                process_check(sample_movie)
                os.chdir(current_path)
            else:
                print(f"Already processed {i}")

def process_check(sample_movie):
    if os.path.isfile('XDS.INP'):
        if os.path.isfile('X-CORRECTIONS.cbf') == False:
            with open('XDS.LP', "w+") as xds_out:
                print("XDS is running...")
                run("xds", stdout= xds_out)
            
        if os.path.isfile('XPARM.XDS') == False:
            for i in range(10):
                with open('XDS.INP', 'r+') as f:
                    lines = f.readlines()
                    f.seek(0)
                    bkgrnd_pix = random.randrange(2, 5, 1)
                    s_pix = random.randrange(4, 9, 1)
                    min_pix = random.randrange(4, 9, 1)
                    f.writelines(lines[:-5]) 
                    f.write(f"""BACKGROUND_PIXEL={bkgrnd_pix}
SIGNAL_PIXEL= {s_pix}  
MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= {min_pix}
!
!""")
                    print("Screening new indexing values.") 
                    with open('XDS.LP', "w+") as xds_out:
                        run("xds", stdout= xds_out)
                if os.path.isfile('XPARM.XDS'):
                    if os.path.isfile('DEFPIX.LP') == False:
                        with open('XDS.INP', 'r+') as f:
                            lines = f.readlines()
                            f.seek(0)
                            f.write(f"""!JOB=XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT
JOB=DEFPIX INTEGRATE CORRECT
""")
                            f.writelines(lines[2:]) 
                            print("Less than 50% of spots went through:")
                            print("Running with JOB= DEFPIX INTEGRATE CORRECT...")
                            with open('XDS.LP', "w+") as xds_out:
                                run("xds", stdout= xds_out) 
                            if os.path.isfile('XPARM.XDS') == False:
                                print(f"Unable to autoprocess {sample_movie}!")
                                break
                                
                            else:
                                return process_check(sample_movie)
                    else:
                        return process_check(sample_movie)
            else:
                print(f"Unable to autoprocess {sample_movie}!")
        
        elif os.path.isfile('DEFPIX.LP') == False:
            with open('XDS.INP', 'r+') as f:
                lines = f.readlines()
                f.seek(0)
                f.write(f"""!JOB=XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT
JOB=DEFPIX INTEGRATE CORRECT
""")
                f.writelines(lines[2:]) 
            print("Less than 50% of spots went through:")
            print("Running with JOB= DEFPIX INTEGRATE CORRECT...")
            with open('XDS.LP', "w+") as xds_out:
                run("xds", stdout= xds_out)
            return process_check(sample_movie)

        elif os.path.isfile("INTEGRATE.HKL") == False:
            with open('XDS.INP', 'r+') as f:
                lines = f.readlines()
                f.seek(0)
                f.writelines(lines) 
                f.write("""
BEAM_DIVERGENCE= 0.03 BEAM_DIVERGENCE_E.S.D.= 0.003
REFLECTING_RANGE=1.0 REFLECTING_RANGE_E.S.D.= 0.2""")
                print("Adding beam divergence values to correct a common error.")
                with open('XDS.INP') as f1:
                    lines = f1.readlines()
                if os.path.isfile("INTEGRATE.HKL") == False:
                    print(f"Unable to autoprocess {sample_movie}!")
                else:
                    return process_check(sample_movie)
        elif os.path.isfile("CORRECT.LP") == True:
            print ("Successful indexing!")
            return mosaicity(sample_movie)

def mosaicity(sample_movie):
    with open('XDS.INP', 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.write("""!JOB=XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT
JOB=DEFPIX INTEGRATE CORRECT
""")
        f.writelines(lines[2:-2])
    with open('INTEGRATE.LP', 'r') as l1, open('XDS.INP', 'a') as f:
        for line in l1:
            if "BEAM_DIVERGENCE= " in line or "REFLECTING_RANGE=" in line:
                f.write(line)
    return iterate_opt(sample_movie)


def iterate_opt(sample_movie):
    with open('XDS.LP') as f1:
        lines = f1.readlines()
    with open('XDS.LP', 'w') as f2:
        f2.writelines(lines[-26:]) 
    with open('XDS.LP', 'r') as f:
        line = f.readline()
        for line in f:
            if ["a", "b", "ISa"] == line.split():
                next_line = f.readline()
                stats = str.split(next_line)
                Isa1 = float(stats[2])
                print(f"Isa: {Isa1}. Testing new values now.")
    with open('XDS.LP', "w+") as xds_out:
        run("xds", stdout= xds_out)
    with open('XDS.LP') as f1:
        lines = f1.readlines()
    with open('XDS.LP', 'w') as f2:
        f2.writelines(lines[-26:]) 
    with open('XDS.LP', 'r') as f:
        line = f.readline()
        for line in f:
            if ["a", "b", "ISa"] == line.split():
                new_next_line = f.readline()
                new_stats = str.split(new_next_line)             
                Isa2 = float(new_stats[2])
                print(f"Isa: {Isa2}")
            if "SPACE_GROUP_NUMBER=" in line:
                number = str.split(line)
                space_group = number[1]
            if "UNIT_CELL_CONSTANTS=" in line:
                cell = str.split(line)
                temp = cell[-6:]
                temp_str = str(temp).strip("]['")
                temp_str2 = temp_str.replace(",","")
                unit_cell = temp_str2.replace("'","")
    Isa_change = abs(Isa2 - Isa1)
    if Isa_change > 0.5:
        print("I'm trying to optimize beam divergence values.")
        return iterate_opt(sample_movie)
    else:
        if space_group is not None and unit_cell is not None:
            print("Optimized beam divergence values.")
            f = open('stats.LP','w')
            f.write(str(space_group) + "\n" + unit_cell)
            f.close()
            print(f"I found space group {space_group} and a unit cell of")
            print(unit_cell)
            return scale_conv(sample_movie)

def scale_conv(sample_movie):
    if os.path.isfile("CORRECT.LP"):
        # Create XSCALE.INP and run xscale
        with open('XSCALE.INP', 'w') as xscale:
            xscale.write(f"""OUTPUT_FILE= {sample_movie}.ahkl
INPUT_FILE= XDS_ASCII.HKL
RESOLUTION_SHELLS= 10 8 5 3 2.3 2.0 1.7 1.5 1.3 1.2 1.1 1.0 0.90 0.80
""")
        with open("xscale.LP", "w+") as xscale_out:
            run("xscale", stdout=xscale_out)
        print("I scaled the data in XSCALE.")
        
        # Create XDSCONV.INP and run xdsconv
        with open('XDSCONV.INP', 'w') as xdsconv:
            xdsconv.write(f"""INPUT_FILE= {sample_movie}.ahkl
OUTPUT_FILE= {sample_movie}.hkl SHELX
GENERATE_FRACTION_OF_TEST_REFLECTIONS=0.10
FRIEDEL'S_LAW=FALSE
""")
        with open("xdsconv.LP", "w+") as xdsconv_out:
            run("xdsconv", stdout=xdsconv_out)
        print("I converted it for use in shelx!")
        
        # Run CCP4's pointless and process its output
        run("/central/groups/NelsonLab/programs/ccp4-8.0/bin/pointless XDS_ASCII.HKL > pointless.LP", shell=True)
        
        # Parse space group information from pointless output
        with open('pointless.LP', 'r') as p1:
            lines = p1.readlines()
            for index, line in enumerate(lines):
                if ["Spacegroup", "TotProb", "SysAbsProb", "Reindex", "Conditions"] == line.split():
                    with open("pointless_group.LP", 'w') as pg:
                        for i in range(2, 7):  # Process lines index+2 to index+6
                            sp_items = lines[index + i].split()
                            for item in sp_items:
                                if item.endswith(")"):
                                    # Extract and process only the first matching item
                                    sp_cleaned = item.strip(')(')
                                    pg.write(f"{sp_cleaned}\n")  # Using f-string for writing to file
                                    # Print out the parsed space group information using f-string
                                    print(f"Possible space group: {sp_cleaned}")
                                    # Break after the first match per line
                                    break

            

if __name__== "__main__":
    process_movie()
