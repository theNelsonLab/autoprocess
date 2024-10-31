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
                     "-o", sample_movie + "_###.img", os.path.join("..", i)],
                     stdout=open(os.devnull, 'wb'))

def process_movie():
    current_path = os.getcwd()
    files = os.listdir()

    s_pix = 7
    min_pix = 7
    bkgrnd_pix = 4
    pix_size = 0.028

    for i in files:
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

                for emi in files:
                    if emi.startswith(f"{sample_movie}_{distance}") and emi.endswith(".emi"):
                        os.rename(emi, os.path.join(movie_path, emi))

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

BACKGROUND_PIXEL= {bkgrnd_pix}
SIGNAL_PIXEL= {s_pix}
MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= {min_pix}
""")

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
        if not os.path.isfile('X-CORRECTIONS.cbf'):
            with open('XDS.LP', "w+") as xds_out:
                print("XDS is running...")
                run("xds", stdout=xds_out)
            
        if not os.path.isfile('XPARM.XDS'):
            for _ in range(10):
                with open('XDS.INP', 'r+') as f:
                    lines = f.readlines()

                bkgrnd_pix = random.randrange(3, 5, 1)
                s_pix = random.randrange(4, 9, 1)
                min_pix = random.randrange(5, 9, 1)

                for index, line in enumerate(lines):
                    if "BACKGROUND_PIXEL=" in line:
                        lines[index] = f"BACKGROUND_PIXEL= {bkgrnd_pix}\n"
                    if "SIGNAL_PIXEL=" in line:
                        lines[index] = f"SIGNAL_PIXEL= {s_pix}\n"
                    if "MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT=" in line:
                        lines[index] = f"MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= {min_pix}\n"

                with open('XDS.INP', 'w') as f:
                    f.writelines(lines)

                print("Screening new indexing values.")
                with open('XDS.LP', "w+") as xds_out:
                    print("XDS is running...")
                    run("xds", stdout=xds_out)

                if os.path.isfile('XPARM.XDS'):
                    if not os.path.isfile('DEFPIX.LP'):
                        with open('XDS.INP', 'r+') as f:
                            lines = f.readlines()
                            f.seek(0)
                            f.write("""!JOB=XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT
JOB=DEFPIX INTEGRATE CORRECT
""")
                            f.writelines(lines[2:])

                        print("Less than 50% of spots went through:")
                        print("Running with JOB= DEFPIX INTEGRATE CORRECT...")
                        
                        with open('XDS.LP', "w+") as xds_out:
                            print("XDS is running...")
                            run("xds", stdout=xds_out)

                        if not os.path.isfile('XPARM.XDS'):
                            print(f"Unable to autoprocess {sample_movie}!")
                            break
                        else:
                            return process_check(sample_movie)
                    else:
                        return process_check(sample_movie)

            print(f"Unable to autoprocess {sample_movie}!")
        
        elif not os.path.isfile('DEFPIX.LP'):
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
                print("XDS is running...")
                run("xds", stdout=xds_out)

            return process_check(sample_movie)

        elif not os.path.isfile("INTEGRATE.HKL"):
            with open('XDS.INP', 'r+') as f:
                lines = f.readlines()
                f.seek(0)
                f.writelines(lines)
                f.write("""BEAM_DIVERGENCE= 0.03 BEAM_DIVERGENCE_E.S.D.= 0.003
REFLECTING_RANGE=1.0 REFLECTING_RANGE_E.S.D.= 0.2""")

            print("Adding beam divergence values to correct a common error.")
                
            with open('XDS.LP', "w+") as xds_out:
                print("XDS is running...")
                run("xds", stdout=xds_out)
            
            if not os.path.isfile("INTEGRATE.HKL"):
                print(f"Unable to autoprocess {sample_movie}!")
            else:
                return process_check(sample_movie)
                
        elif os.path.isfile("CORRECT.LP"):
            print ("Successful indexing!")
            return mosaicity(sample_movie)


def mosaicity(sample_movie):
    with open('XDS.INP', 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.write("""!JOB=XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT
JOB=DEFPIX INTEGRATE CORRECT
""")
        f.writelines(lines[2:])

    beam_divergence = None
    reflecting_range = None

    with open('INTEGRATE.LP', 'r') as l:
        for line in l:
            if "BEAM_DIVERGENCE=" in line:
                beam_divergence = line.strip()
            if "REFLECTING_RANGE=" in line:
                reflecting_range = line.strip()

    beam_divergence_found = False
    reflecting_range_found = False

    for index, line in enumerate(lines):
        if "BEAM_DIVERGENCE=" in line:
            lines[index] = f"{beam_divergence}\n"
            beam_divergence_found = True
        if "REFLECTING_RANGE=" in line:
            lines[index] = f"{reflecting_range}\n"
            reflecting_range_found = True

    if not beam_divergence_found and beam_divergence:
        lines.append(f"{beam_divergence}\n")
    if not reflecting_range_found and reflecting_range:
        lines.append(f"{reflecting_range}\n")

    with open('XDS.INP', 'w') as f:
        f.writelines(lines)

    return iterate_opt(sample_movie)

def iterate_opt(sample_movie):
    Isa1 = None
    Isa2 = None
    space_group = None
    unit_cell = None

    with open('XDS.LP', 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.writelines(lines[-26:])

    for line in lines:
        if ["a", "b", "ISa"] == line.split():
            next_line = lines[lines.index(line) + 1]
            stats = next_line.split()
            Isa1 = float(stats[2])
            print(f"Isa: {Isa1}. Testing new values now.")
    
    with open('XDS.LP', "w+") as xds_out:
        print("XDS is running...")
        run("xds", stdout=xds_out)
    
    with open('XDS.LP', 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.writelines(lines[-26:])
    
    for line in lines:
        if ["a", "b", "ISa"] == line.split():
            new_next_line = lines[lines.index(line) + 1]
            new_stats = new_next_line.split()
            Isa2 = float(new_stats[2])
            print(f"Isa: {Isa2}")
        if "SPACE_GROUP_NUMBER=" in line:
            space_group = line.split()[1]
        if "UNIT_CELL_CONSTANTS=" in line:
            cell = line.split()[-6:]
            unit_cell = " ".join(cell)

    Isa_change = abs(Isa2 - Isa1)

    if Isa_change > 0.5:
        print("I'm trying to optimize beam divergence values.")
        return iterate_opt(sample_movie)
    else:
        if space_group and unit_cell:
            print("Optimized beam divergence values.")
            with open('stats.LP', 'w') as f:
                f.write(f"{space_group}\n{unit_cell}")
            print(f"I found space group {space_group} and a unit cell of")
            print(unit_cell)
            return scale_conv(sample_movie)
        else:
            print(f"Space group or unit cell not found. Cannot finish autoprocess for {sample_movie}.")

def scale_conv(sample_movie):
    if os.path.isfile("CORRECT.LP"):
        with open('XSCALE.INP', 'w') as xscale:
            xscale.write(f"""OUTPUT_FILE= {sample_movie}.ahkl
INPUT_FILE= XDS_ASCII.HKL
RESOLUTION_SHELLS= 10 8 5 3 2.3 2.0 1.7 1.5 1.3 1.2 1.1 1.0 0.90 0.80
""")
        with open("xscale.LP", "w+") as xscale_out:
            run("xscale", stdout=xscale_out)
        print("I scaled the data in XSCALE.")
        
        with open('XDSCONV.INP', 'w') as xdsconv:
            xdsconv.write(f"""INPUT_FILE= {sample_movie}.ahkl
OUTPUT_FILE= {sample_movie}.hkl SHELX
GENERATE_FRACTION_OF_TEST_REFLECTIONS=0.10
FRIEDEL'S_LAW=FALSE
""")

    return check_space_group(sample_movie)

def check_space_group(sample_movie):
    with open("xdsconv.LP", "w+") as xdsconv_out:
        run("xdsconv", stdout=xdsconv_out)
    print("I converted it for use in shelx!")

    # Run CCP4's pointless and process its output
    run("/central/groups/NelsonLab/programs/ccp4-8.0/bin/pointless XDS_ASCII.HKL > pointless.LP",
        shell=True)

    with open('pointless.LP', 'r') as p1:
        lines = p1.readlines()
        for index, line in enumerate(lines):
            if ["Spacegroup", "TotProb", "SysAbsProb", "Reindex", "Conditions"] == line.split():
                with open("pointless_group.LP", 'w') as pg:
                    for i in range(2, 7):  # Process lines index+2 to index+6
                        sp_items = lines[index + i].split()
                        for item in sp_items:
                            if item.endswith(")"):
                                sp_cleaned = item.strip(')(')
                                pg.write(f"{sp_cleaned}\n")
                                print(f"Possible space group: {sp_cleaned}")
                                break



if __name__== "__main__":
    print(r"")
    print(r"    ___         __       ____                               ")
    print(r"   /   | __  __/ /_____ / __ \___________________________   ")
    print(r"  / /| |/ / / / __/ __ / /_/ / __/__ / __/ _  / ___/ ___/hmn")
    print(r" / ___ / /_/ / /_/ /_// /\__/ // /_// /_/  __(__  (__  )jeb ")
    print(r"/_/  |_\____/\__/\___/_/   /_/ \___/\___/\__/\___/\___/dbe  ")
    print(r"")
    process_movie()
