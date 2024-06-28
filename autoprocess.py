#
"""
autoprocess_version2 original file from Jessica Burch
"""
import os
from subprocess import run
import random
import string

def convert():
    file = os.listdir()
    for i in file:
        if i.endswith(".ser"):
            #This assumes you followed the naming convention of "name_distance_rotation_exposure.ser"
            split = i.split("_")
            if split[0] not in file:
                os.mkdir(split[0])
                os.rename(i, split[0] + "/" + i)
                os.mkdir(split[0] + "/images")
                os.mkdir(split[0] + "/auto_process")
                os.chdir(split[0] + "/images")
                # P = pixel size, B = binning, r = rotation rate, w = wavelength in A, d = detector distance in mm, E = exposure time, M = offset
                run(["/groups/NelsonLab/programs/ser2smv", "-P", "0.014", "-B", "2", "-r", split[2], "-w", "0.0251", "-d", split[1], "-E", split[3], "-M", "200", "-v", "-o", split[0] + "_###.img", "../" + i], stdout=open(os.devnull, 'wb'))
                path = os.getcwd()
                os.chdir("../auto_process")
                image_number = str(len(os.listdir("../images")))
                spot = str(4)
                min_pix = str(7)
                xds_inp = open("XDS.INP", "w")
                data_path = str(path + "/" + split[0])
                xds_inp.write("JOB= XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT" +
                            "\n!JOB=DEFPIX INTEGRATE CORRECT" +
                            "\nORGX= 1018 ORGY= 1008 ! check these using adxv" + # X and Y of beam center
                            "\nDETECTOR_DISTANCE= " + str(split[1]) +
                            "\nOSCILLATION_RANGE= " + str(float(split[3]) * float(split[2])) + "\nX-RAY_WAVELENGTH= 0.0251000002" + # wavelength in A
                            "\n\nNAME_TEMPLATE_OF_DATA_FRAMES=" + data_path + "_???.img" + 
                            "\nBACKGROUND_RANGE=1 10\n!DELPHI=15\n!SPACE_GROUP_NUMBER=0" +
                            "\n!UNIT_CELL_CONSTANTS= 1 1 1 90 90 90" + "\nINCLUDE_RESOLUTION_RANGE= 40 " + str(((float(split[1]) * 0.0009) - 0.1)) +
                            "\nTEST_RESOLUTION_RANGE= 40 " + str(round(((float(split[1]) * 0.0009) - 0.1),ndigits=2)) + "\nTRUSTED_REGION=0.0 1.2"+ 
                            "\nVALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS=6000. 30000. ! parameters for detector and beamline:" +
                            "\nDETECTOR= ADSC MINIMUM_VALID_PIXEL_VALUE= 1 OVERLOAD= 65000" + 
                            "\nSENSOR_THICKNESS= 0.01\nNX= 2048 NY= 2048 QX= 0.0280000009" + # detector parameters
                            " QY= 0.0280000009\nROTATION_AXIS=0 -1 0" + # !!! this is a common reason data doesn't index, check for each scope
                            "\nDIRECTION_OF_DETECTOR_X-AXIS=1 0 0" + 
                            "\nDIRECTION_OF_DETECTOR_Y-AXIS=0 1 0" +
                            "\nINCIDENT_BEAM_DIRECTION=0 0 1\nFRACTION_OF_POLARIZATION=0.98"
                            + "\nPOLARIZATION_PLANE_NORMAL=0 1 0" +
                            "\nREFINE(IDXREF)=CELL BEAM ORIENTATION AXIS ! DISTANCE" +
                            "\nREFINE(INTEGRATE)= DISTANCE BEAM ORIENTATION ! AXIS CELL" +
                            "\nREFINE(CORRECT)=CELL BEAM ORIENTATION AXIS ! DISTANCE !" + 
                            "\n\nDATA_RANGE= 1 " + image_number + "\nSPOT_RANGE= 1 " + image_number +
                            "\nSTRONG_PIXEL= " + spot + "\nMINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= " + min_pix + "\n!\n!")
                xds_inp.close()
                xds_out = open("XDS.LP", "w+")
                print("Processing " + split[0] + "...")
                run("xds", stdout=xds_out)
                process_check()
                os.chdir('../../')
            else:
                print("Already processed " + i)

def process_check():
    if os.path.isfile('XDS.INP') == True:
        if os.path.isfile('X-CORRECTIONS.cbf') == False:
            xds_out = open("XDS.LP", "w+")
            print("XDS is running...")
            run("xds", stdout= xds_out)
            
        if os.path.isfile('XPARM.XDS') == False:
            for i in range(10):
                with open('XDS.INP') as f1:
                    lines = f1.readlines()
                with open('XDS.INP', 'w') as f2:
                    strong = random.randrange(3,9,1)
                    mpix = random.randrange(4,9,1)
                    f2.writelines(lines[:-4]) 
                    f2.write("STRONG_PIXEL= " + str(strong) +  
                            "\nMINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= " + str(mpix) +
                            "\n!\n!")
                    f2.close()
                    print("Screening new indexing values.") 
                    xds_out = open("XDS.LP", "w+")
                    run("xds",stdout= xds_out)
                if os.path.isfile('XPARM.XDS') == True:
                    if os.path.isfile('DEFPIX.LP') == False:
                        with open('XDS.INP') as f1:
                            lines = f1.readlines()
                        with open('XDS.INP', 'w') as f2:
                            f2.write("!JOB=XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT" 
                                    + "\nJOB=DEFPIX INTEGRATE CORRECT\n")
                            f2.writelines(lines[2:]) 
                            f2.close()
                            print("Less than 70% of spots went through. Running with JOB= DEFPIX "
                                + "INTEGRATE CORRECT.")
                            xds_out = open("XDS.LP", "w+")
                            run("xds",stdout= xds_out) 
                            if os.path.isfile('XPARM.XDS') == False:
                                name = os.getcwd().split("/")[-2]
                                print("Unable to autoprocess " + name + "!")
                                break
                                
                            else:
                                return process_check()
                    else:
                        return process_check()
            else:
                name = os.getcwd().split("/")[-2]
                print("Unable to autoprocess " + name + "!")
                f2.close()
                

        
        elif os.path.isfile('DEFPIX.LP') == False:
            with open('XDS.INP') as f1:
                lines = f1.readlines()
            with open('XDS.INP', 'w') as f2:
                f2.write("!JOB=XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT" 
                        + "\nJOB=DEFPIX INTEGRATE CORRECT\n")
                f2.writelines(lines[2:]) 
                f2.close()
            print("Less than 70% of spots went through. Running with JOB= DEFPIX "
                + "INTEGRATE CORRECT.")
            xds_out = open("XDS.LP", "w+")
            run("xds",stdout= xds_out)
            return process_check()

        elif os.path.isfile("INTEGRATE.HKL") == False:
            with open('XDS.INP') as f1:
                lines = f1.readlines()
            with open('XDS.INP', 'w') as f2:
                f2.writelines(lines) 
                f2.write("\nBEAM_DIVERGENCE= 0.03 BEAM_DIVERGENCE_E.S.D.= 0.003" +
                "\nREFLECTING_RANGE=1.0 REFLECTING_RANGE_E.S.D.= 0.2")
                f2.close()
                print("Adding beam divergence values to correct a common error.")
                xds_out = open("XDS.LP", "w+")
                run("xds",stdout= xds_out)
                if os.path.isfile("INTEGRATE.HKL") == False:
                    name = os.getcwd().split("/")[-2]
                    print("Unable to autoprocess " + name + "!")
                else:
                    return process_check()
        elif os.path.isfile("CORRECT.LP") == True:
            print ("Successful indexing!")
            return mosaicity()

def mosaicity():
    with open('XDS.INP') as f1:
        lines = f1.readlines()
    with open('XDS.INP', 'w') as f2:
        f2.write("!JOB=XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT" 
                 + "\nJOB=DEFPIX INTEGRATE CORRECT\n")
        f2.writelines(lines[2:-2]) 
    with open('INTEGRATE.LP', 'r') as l1:
        f2 = open('XDS.INP', 'a')
        line = l1.readline()
        for line in l1:
            if "BEAM_DIVERGENCE= " in line:
                f2.write(line)
            if "REFLECTING_RANGE=" in line:
                f2.write(line)
        f2.close()
        return iterate_opt()

def iterate_opt():
    with open('XDS.LP') as f1:
        lines = f1.readlines()
    with open('XDS.LP', 'w') as f2:
        f2.writelines(lines[-26:]) 
    with open('XDS.LP', 'r') as f:
        line = f.readline()
        for line in f:
            if "     a        b          ISa" in line:
                next_line = f.readline()
                stats = str.split(next_line)
                Isa1 = float(stats[2])
                print("Isa: " + str(Isa1) + ". Testing new values now.")
    xds_out = open("XDS.LP", "w+")
    run("xds",stdout= xds_out)
    with open('XDS.LP') as f1:
        lines = f1.readlines()
    with open('XDS.LP', 'w') as f2:
        f2.writelines(lines[-26:]) 
    with open('XDS.LP', 'r') as f:
        line = f.readline()
        for line in f:
            if "     a        b          ISa" in line:
                new_next_line = f.readline()
                new_stats = str.split(new_next_line)             
                Isa2 = float(new_stats[2])
                print("Isa: " + str(Isa2))
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
        return iterate_opt()
    else:
        if space_group and unit_cell != None:
            print("Optimized beam divergence values.")
            f = open('stats.LP','w')
            f.write(str(space_group) + "\n" + unit_cell)
            f.close
            print("I found space group " + str(space_group) + " and a unit cell of "
                + "\n" + unit_cell)
            return scale_conv()


def scale_conv():
    if os.path.isfile("CORRECT.LP") == True:
        xscale = open('XSCALE.INP','w')
        xscale_out = open("xscale.LP","w+")
        m = os.getcwd().split("/")[-2]
        xscale.write("OUTPUT_FILE= " + m +".ahkl"+"\nINPUT_FILE= XDS_ASCII.HKL"
                    + "\nRESOLUTION_SHELLS= 10 8 5 3 2.3 2.0 1.7 1.5 1.3 " + 
                    "1.2 1.1 1.0 0.90 0.80")
        xscale.close()
        run("xscale", stdout= xscale_out)
        print("I scaled the data in XSCALE.")
        xdsconv_out = open("xdsconv.LP", "w+")
        xdsconv = open('XDSCONV.INP','w')
        xdsconv.write("INPUT_FILE= " + m + ".ahkl" + "\nOUTPUT_FILE= " + 
                    m + ".hkl" + " SHELX" +
                    "\nGENERATE_FRACTION_OF_TEST_REFLECTIONS=0.10"
                    + "\nFRIEDEL'S_LAW=FALSE")
        xdsconv.close()
        run("xdsconv",stdout= xdsconv_out)
        print("I converted it for use in shelx!")
        os.system("/central/groups/NelsonLab/programs/ccp4-8.0/bin/pointless XDS_ASCII.HKL > pointless.LP")
        with open('pointless.LP','r') as p1:
            lines = p1.readlines()
            for index, line in enumerate(lines):
                if "   Spacegroup         TotProb SysAbsProb     Reindex         Conditions" in line:
                    os.mknod("pointless_group.LP")
                    sp1 = lines[index+2]
                    sp1_1 = sp1.split()
                    for item in sp1_1:
                        if item.endswith(")") == True:
                            sp1_2 = str(item).strip(')(')
                            with open('pointless_group.LP','a') as pg:
                                pg.write(str(sp1_2) + "\n")
                    sp2 = lines[index+3]
                    sp2_1 = sp2.split()
                    for item in sp2_1:
                        if item.endswith(")") == True:
                            sp2_2 = str(item).strip(')(')
                            with open('pointless_group.LP','a') as pg:
                                pg.write(str(sp2_2) + "\n")
                    sp3 = lines[index+4]
                    sp3_1 = sp3.split()
                    for item in sp3_1: 
                        if item.endswith(")"):
                            sp3_2 = str(item).strip(')(')
                            with open('pointless_group.LP','a') as pg:
                                pg.write(str(sp3_2) + "\n")
                    sp4 = lines[index+5]
                    sp4_1 = sp4.split()
                    for item in sp4_1: 
                        if item.endswith(")"):
                            sp4_2 = str(item).strip(')(')
                            with open('pointless_group.LP','a') as pg:
                                pg.write(str(sp4_2) + "\n")
                    sp5 = lines[index+6]
                    sp5_1 = sp5.split()
                    for item in sp5_1: 
                        if item.endswith(")"):
                            sp5_2 = str(item).strip(')(')
                            with open('pointless_group.LP','a') as pg:
                                pg.write(str(sp5_2) + "\n")
            

if __name__== "__main__":
    convert()
