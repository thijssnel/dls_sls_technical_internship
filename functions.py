import urllib
from bs4 import BeautifulSoup
import requests
import os, shutil, subprocess
import urllib
import requests
import io
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
from jupyter_analysis_tools.utils import (isWindows, isMac, isLinux,
    isList, pushd, grouper, updatedDict)
import sys

import os



InputFn = "contin_in.txt"
OutputFn = "contin_out.txt"



def angleToQ(deg, refrac, wavelen):
    return 4.*np.pi*refrac/wavelen*np.sin(deg*np.pi/360.)

# constant for the experiment at each angle
def calcDLSconst(temp, visc):
    kB = 1.38066E-23 # Boltzman constant, [kB] = Newton m / Kelvin
    return kB*temp/(6*np.pi*visc)

def getDLSgammaSi(angle, refrac, wavelen, temp, visc):
    return calcDLSconst(temp, visc) * angleToQ(angle,refrac, wavelen)**2


def workerInit(_queue):
    """Initializes a queue for log messages in each worker process during multiprocessing.
    Queue object is global within each process only, not in the parent."""
    global queue
    queue = _queue



def getContinOnline(targetPath, binaryName):
    baseurl="http://www.s-provencher.com"
    html_page = urllib.request.urlopen(baseurl+"/contin.shtml").read()
    soup = BeautifulSoup(html_page)
    binurl = [link.get('href') for link in soup.findAll('a') if link.text.strip() == binaryName]
    binurl = '/'.join((baseurl,binurl[0]))
    binary = requests.get(binurl, allow_redirects=True)
    open(targetPath, 'wb').write(binary.content)
    return targetPath

def getContinPath():
    if isMac():
        # get local path to the CONTIN executable
        continCmd = Path.home() / "code" / "cntb2" / "bin" / "contin OSX"
    elif isWindows():
        continCmd = Path(os.getenv('APPDATA')) / "contin" / "contin.exe"
        if continCmd.is_file():
            return continCmd
        continCmd.parent.mkdir(parents=True, exist_ok=True)
        getContinOnline(continCmd, "contin-windows.exe")
    elif isLinux():
        continCmd = Path.home() / ".local" / "bin" / "contin"
        if continCmd.is_file():
            return continCmd
        continCmd.parent.mkdir(parents=True, exist_ok=True)
        getContinOnline(continCmd, "contin-linux")
        if continCmd.is_file():
            continCmd.chmod(0o755) # make it executable
    if continCmd.is_file():
        print(f"Installed CONTIN at '{continCmd}'.")
        return continCmd
    print(f"Failed to find CONTIN at '{continCmd}'!")
    raise NotImplementedError("Don't know how to retrieve the CONTIN executable!")

def genContinInput(filedata, continConfig):
    """Generate a CONTIN input file (bytes) for a single dataset.

    Assumptions / conventions:
      - continConfig['fitRangeM'] is (r_min_m, r_max_m) in meters (hydrodynamic radius).
      - filedata contains:
          .Angle_deg (degrees)
          .Temperature_K (K)
          .Viscosity_cp (centipoise, i.e. mPa·s)
          .Refractive_Index (dimensionless)
          .Wavelength_nm (nm)
          .Correlationx (tau in seconds)
          .Correlationy (g2 values, not transformed)
      - getContinUserVars() expects RUSER 19 in centipoise (mPa·s) and RUSER 16 in nm.
    """
    # IWT: noise weighting used by CONTIN (5 for photon statistics, 1 otherwise)
    IWT = 5 if continConfig.get('weighResiduals', False) else 1

    # Transform flag (keep same convention you used before)
    Trd = -1  # keep previous behavior (your repo used -1)

    # Measurement metadata
    angle = float(filedata.Angle_deg)
    temp = float(filedata.Temperature_K)        # K
    visc_cp = float(filedata.Viscosity_cp)     # centipoise (mPa·s)
    refrac = float(filedata.Refractive_Index)
    wavelen_nm = float(filedata.Wavelength_nm) # nm

    # --- convert to SI for calculations ---
    visc_Pa_s = visc_cp * 1e-3      # mPa.s (centipoise) -> Pa.s
    wavelen_m = wavelen_nm * 1e-9   # nm -> m

    # compute q (uses same formula you used elsewhere)
    q = angleToQ(angle, refrac, wavelen_m)  # this returns 4π n / λ * sin(θ/2)

    # calc constant gamma_unit = (kB*T)/(6π η) * q^2
    # note: gamma_unit gives Gamma for R = 1 m. For radius r (m): Gamma(r) = gamma_unit / r
    kB = 1.38066E-23
    calc_const = kB * temp / (6.0 * np.pi * visc_Pa_s)
    gamma_unit = calc_const * (q**2)  # [1/s] * m  (for R=1 m -> gives 1/s)

    # --- compute the decay-rate search range (GMNMX) from radius fitRangeM ---
    rmin_m, rmax_m = continConfig.get('fitRangeM', (1e-9, 1e-6))
    # sanity: ensure rmin < rmax
    if rmin_m <= 0 or rmax_m <= 0 or rmin_m >= rmax_m:
        raise ValueError(f"Invalid fitRangeM: {continConfig.get('fitRangeM')}")

    # decay rates: Gamma_min = gamma_unit / rmax  ;  Gamma_max = gamma_unit / rmin
    Gamma_min = rmax_m/gamma_unit 
    Gamma_max = rmin_m/gamma_unit 

    # Safety: avoid zero or negative values
    if not np.isfinite(Gamma_min) or not np.isfinite(Gamma_max) or Gamma_min <= 0 or Gamma_max <= 0:
        raise RuntimeError(f"Computed invalid Gamma_min/Gamma_max: {Gamma_min}, {Gamma_max}")

    # For a robust CONTIN grid it's sometimes helpful to expand the gamma range slightly:
    gamma_pad_factor = continConfig.get('gammaPadFactor', 1.0)  # default no pad
    if gamma_pad_factor != 1.0:
        center = np.sqrt(Gamma_min * Gamma_max)
        Gamma_min = center / gamma_pad_factor
        Gamma_max = center * gamma_pad_factor

    # Debug print (helpful during development)
    print(f"DEBUG CONTIN input: angle={angle}°, T={temp}K, visc={visc_cp} cp, λ={wavelen_nm} nm")
    print(f"DEBUG q={q:.3e}  calc_const={calc_const:.3e} gamma_unit={gamma_unit:.3e}")
    print(f"DEBUG Gamma_min={Gamma_min:.3e}  Gamma_max={Gamma_max:.3e}")

    # get measured correlation data and tau (restrict to ptRangeSec)
    dlsDatay = np.asarray(filedata.Correlationy, dtype=float)
    dlsDatax = np.asarray(filedata.Correlationx, dtype=float)

    tmin, tmax = continConfig.get('ptRangeSec', (np.min(dlsDatax), np.max(dlsDatax)))
    tmask = np.logical_and(tmin <= dlsDatax, dlsDatax <= tmax)
    tauCropped = dlsDatax[tmask]
    corCropped = dlsDatay[tmask]

    if tauCropped.size == 0:
        raise RuntimeError("No tau points left after applying ptRangeSec. Check units and ranges.")

    # CONTIN expects the input transformation: many DLS pipelines feed sqrt(g2-1)
    # You set Trd = -1 above (which in your repo meant input is g2-1?)
    # To be safe, write the correlation values exactly as they were cropped; CONTIN transform flag
    # is left at Trd so CONTIN will interpret them accordingly.
    # Ensure values are finite
    tauCropped = np.asarray(tauCropped, dtype=float)
    corCropped = np.asarray(corCropped, dtype=float)
    finite_mask = np.isfinite(tauCropped) & np.isfinite(corCropped)
    tauCropped = tauCropped[finite_mask]
    corCropped = corCropped[finite_mask]

    if tauCropped.size == 0:
        raise RuntimeError("All cropped tau/correlation values are NaN or Inf.")

    # format arrays for CONTIN (6E13.7 style; we emulate with exponential format)
    a2s_kwargs = dict(floatmode='fixed', sign=' ', max_line_width=80,
                      formatter={'float_kind': '{0: .5E}'.format})
    tauStr = np.array2string(tauCropped, **a2s_kwargs)[1:-1]
    corStr = np.array2string(corCropped, **a2s_kwargs)[1:-1]
    npts = len(tauCropped)

    storedFn = filedata.Path.name
    if not filedata.Path.is_file():
        try:
            storedFn = filedata['filename'].parent.suffix + '/' + storedFn
        except Exception:
            # fallback: leave storedFn as name
            pass

    # prepare numeric fields to write (RUSER expects viscosity in centipoise and wavelength in nm)
    # filedata already stores visc as centipoise and wavelength as nm in your class; use those
    visc_for_ruser = visc_cp          # centipoise (mPa.s)
    wavelen_for_ruser = wavelen_nm    # nm

    # grid and baseline from config
    gridpts = int(continConfig.get('gridpts', 200))
    baselineCoeffs = int(continConfig.get('baselineCoeffs', 0))

    # build the content string (no .format mixing; use f-string with explicit variables)
    content = f"""{storedFn}
 IFORMY    0    .00
 (6E13.7)
 IFORMT    0    .00
 (6E13.7)
 IFORMW    0    .00
 (6E13.7)
 NINTT     0   -1.00
 DOUSNQ    0    1.00
 IQUAD     0    1.00
 PRWT      0    1.00
 PRY       0    1.00
 IPLRES    1    3.00
 IPLRES    2    3.00
 IPRINT    1    0.00
 IPRINT    2    2.00
 IPLFIT    1    0.00
 IPLFIT    2    0.00
 LINEPG    0   50.
 NONNEG    1    1.00
 IWT       0    {IWT:.2f}
 NQPROG    1    5.00
 NQPROG    2    5.00
 GMNMX     1    {Gamma_min:.5E}
 GMNMX     2    {Gamma_max:.5E}
 RUSER    10    {Trd:.2f}
 NG        0    {gridpts:.2f}
 NLINF     0    {baselineCoeffs:.2f}
 IUSER    10    4.00
 RUSER    21    1.0
 RUSER    22   -1.0
 RUSER    23    0.0
 RUSER    18         {temp:.5f}
 RUSER    17         {angle:.5f}
 RUSER    19         {visc_for_ruser:.5f}
 RUSER    15         {refrac:.5f}
 RUSER    16         {wavelen_for_ruser:.5f}
 RUSER    25         {1}
 RUSER    26         {0}
 END       0    0.00
 NY{npts: >9d}
 {tauStr}
 {corStr}
""".format(**continConfig
    )
    return content.encode('ascii')


def runContin(filedata, continConfig, useQueue=True):
    """Starts a single CONTIN process for the given DLS DataSet
    (which should contain a single angle only)."""
    name = filedata[0]
    filedata = filedata[1]
    continCmd = getContinPath()
    assert continCmd.is_file(), "CONTIN executable not found!"
    logPrefix = f"{filedata.Path.name}@{filedata.Angle_deg}°: "
    workDir = filedata.Path.parent
    if workDir.is_file():
        logPrefix = workDir.stem + '/' + logPrefix
        workDir = workDir.parent / workDir.stem
    logFunc = queue.put if useQueue else print
    def log(text):
        #print("="+logPrefix+text)
        logFunc(" "+logPrefix+text)
    try:
        continInData = genContinInput(filedata, continConfig)
##########################################
    except AssertionError:
        log(f"Scattering angle  not found! "
            f"Skipping…")
        return
    #ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # timestamp
    tmpDir = workDir / ('Contin'+'_'+str(name))
    # if tmpDir.is_dir(): # deleting old results
    #     if not continConfig.get("recalc", True):
    #         return tmpDir
    #     shutil.rmtree(tmpDir)
    if tmpDir.is_dir(): # deleting old results
        if not continConfig.get("recalc", True):
            return tmpDir
        shutil.rmtree(tmpDir)
    os.mkdir(tmpDir)
    continInDataPath  = tmpDir / InputFn
    continOutDataPath = tmpDir / OutputFn

    
    # Store input data
    with open(continInDataPath, 'wb') as fd:
        fd.write(continInData)
    with pushd(tmpDir):
        proc = subprocess.run([str(continCmd)], input=continInData,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if len(proc.stderr):
            log(proc.stderr.decode().strip())
    # Store output data
    with open(continOutDataPath, 'wb') as fd:
        fd.write(proc.stdout)
    return tmpDir

def runContinOverFiles(fnLst, configLst, nthreads=None, outputCallback=None):
    """*fnLst*: List of file paths to .ASC files
       *configLst*: List of parameters, one dict for each file, such as
            {'recalc': True, 'gridpts': 200, 'transformData': True,
             'ptRangeSec': (3e-07, 1.0), 'fitRangeM': (7e-10, 3.9e-07),
              'baselineCoeffs': 0, 'weighResiduals': True}
       *nthreads*: number of parallel CONTIN processes to launch,
           1: Sequential processing, one file after another
           None: number of processes equals the number of computing cores
       *outputCallback*: A function with one argument to called repeatedly (0.5s)
           with new output messages combined from all CONTIN processes.
    """
    start = time.time()
    # make sure the contin cmd exists, avoids downloading/installing it from parallel threads later
    continCmd = getContinPath()
    assert continCmd.is_file(), "CONTIN executable not found!"
    # dataLst = readData(fnLst, configLst)
    # get all combinations of CONTIN parameters and data files
    # rootPath = fnLst[0].Path.parent
    # delete_contin_files(rootPath)
    if type(fnLst) == list:
        dataNConfig = [(data, configLst) for data in fnLst]
    else:
        dataNConfig = [(fnLst,configLst)]

    
    if nthreads == 1:
        resultDirs = [runContin(data, cfg, False) for data, cfg in dataNConfig]
    else: # Using multiple CPU cores if available
        import multiprocessing
        if not nthreads:
            nthreads = multiprocessing.cpu_count()
        from multiprocessing import Queue as MPQueue
        # use a queue to collect stdout messages from subprocesses
        logQueue = MPQueue()
        pool = multiprocessing.Pool(processes=nthreads, initializer=workerInit, initargs=(logQueue,))
        resultDirs = pool.starmap_async(runContin, dataNConfig)
        pool.close()
        def resultReady(asyncResult): # checks if the overall result is ready
            try:
                asyncResult.successful()
            except ValueError:
                return False
            return True
        outputBuffer = [] # buffer to store output messages from queue in,
                          # for sorting, for deterministic testing
        while not resultReady(resultDirs):
            time.sleep(.5) # update interval of output
            while not logQueue.empty():
                newOutput = logQueue.get_nowait()
                if not outputCallback:
                    print(newOutput) # the traditional way
                else:
                    outputBuffer.extend(newOutput.splitlines())
            if callable(outputCallback):
                # use a custom callback to handle the output from subprocesses
                outputCallback("\n".join(sorted(outputBuffer)))
        #print("READY!")
        resultDirs = resultDirs.get()

    summary = f"CONTIN analysis with {nthreads} thread{'s' if nthreads > 1 else ''} took {time.time()-start:.1f}s."
    return [rd for rd in resultDirs if rd is not None], summary

def getContinInputCurve(inputAbsPath):
    assert inputAbsPath.is_file()
    # read in line by line, some adjustments required for parsing floats
    startLine, count = 0, 0
    with open(inputAbsPath) as fd:
        startLine, count = [(idx, int(line.split()[-1]))
                            for idx, line in enumerate(fd) if "NY" in line][0]
    lines = []
    with open(inputAbsPath) as fd:
        lines = fd.readlines()
    return [float(f) for line in lines[startLine+1:] for f in line.split()][count:]

def getLineNumber(lines, phrases, debug=False):
    """Returns the line numbers containing the provided phrases after searching
    for the previous phrases sequentially. Search starts with the first phrase,
    once it is found, search starts with the 2nd phrase from that line,
    until the last phrase is found. Ignores early matches of the final phrase."""
    nums = []
    for i, line in enumerate(lines):
        if phrases[len(nums)] in line:
            if debug:
                print("found '{}' on line {}.".format(phrases[len(nums)], i))
            nums.append(i)
            if len(phrases) == len(nums):
                return nums
    return nums

def getValueDictFromLines(lines, **kwargs):
    """Searches the given list of lines for the keys of the given arguments
    and converts the values to float. Returns the completed dict."""
    # search begin of common variables
    lstart = [idx for idx, line in enumerate(lines)
                  if "INPUT DATA FOR CHANGES TO COMMON VARIABLES" in line]
    # search end of common variables section (where the next begins)
    lend   = [idx for idx, line in enumerate(lines)
                  if "FINAL VALUES OF CONTROL VARIABLES" in line]
    result = dict()
    if len(lstart) and len(lend):
        result = {key: float(line.split()[-1])
                    for line in lines[lstart[0]:lend[0]]
                    for key, pattern in kwargs.items() if pattern in line}
    return result

def getContinUserVars(lines):
    """Extract previously set user variables for environmental values
    from CONTIN output data.
    *lines*: List of lines of CONTIN output data."""
    varmap = getValueDictFromLines(lines,
                temp="RUSER    18", angle="RUSER    17", visc="RUSER    19",
                refrac="RUSER    15", wavelen="RUSER    16", score="RUSER    11")
    # convert to SI units
    varmap["visc"] *= 1e-3
    varmap["wavelen"] *= 1e-9
    print(varmap["angle"], varmap["refrac"], varmap["wavelen"],
                                    varmap["temp"], varmap["visc"])
    varmap["gamma"] = getDLSgammaSi(varmap["angle"], varmap["refrac"], varmap["wavelen"],
                                    varmap["temp"], varmap["visc"])
    return varmap

def getContinResults(sampleDir):
    """*sampleDir*: A pathlib Path of the location where the CONTIN results can be found."""
    sampleDir = Path(sampleDir)
    # check first if there was any CONTIN output generated
    resultsFile = sampleDir / OutputFn

    # read in line by line, some adjustments required for parsing floats
    lines = []
    with open(resultsFile) as fd:
        lines = fd.readlines()
    # find the beginning and end of the fitted correlation curve
    startLines = getLineNumber(lines, ["T            Y", "0PRECIS"])
    if not len(startLines):
        print(f"Fitted curve not found in CONTIN output!\n ({resultsFile})")
        return None, None, None
    dfStart, dfEnd = startLines[-2]+1, startLines[-1]
    dfFit = pd.DataFrame([f for line in lines[dfStart:dfEnd] for f in grouper(line.split(), 2)],
                         columns=('tau', 'corrFit'), dtype=float)
    dfFit.corrFit = dfFit.corrFit**2 # to be compared with measured data
    # get input correlation curve first, to be added to fitted correlation curve
    dfFit['corrIn'] = getContinInputCurve(sampleDir/InputFn)
    # find the beginning and end of the distribution data
    startLines = getLineNumber(lines, ["CHOSEN SOLUTION", "ORDINATE"])
    if not len(startLines):
        print(f"Distribution data not found in CONTIN output!\n ({resultsFile})")
        return None, None, None
    gridSize = int(getValueDictFromLines(lines, distribSize="NG        0").get('distribSize',0))
    dfStart = startLines[1]+1
    lineEnd = 31 # do not parse floats beyond this column
    # convert CONTIN output distrib to parseable data for pandas
    fixedFloatFmt = io.StringIO("\n".join([line[:lineEnd].replace("D", "E")
                                for line in lines[dfStart:dfStart+gridSize]]))
    dfDistrib = pd.read_csv(fixedFloatFmt, sep='\s+', names=("distrib", "err", "decay"))
    dfDistrib = dfDistrib[["decay", "distrib", "err"]] # reorder to (x,y,u)
    # update x/abscissa with values from another section of the output
    # to avoid duplicates due to low precision in solution output parsed above
    startLines = getLineNumber(lines, ["GRID POINT"])
    if len(startLines):
        dfStart = startLines[0]+1
        decayNew = np.fromiter([line.split()[0] for line in lines[dfStart:dfStart+gridSize]], float)
        dfDistrib.decay = decayNew
    varmap = getContinUserVars(lines)
    # parse original input data filename as well, if available
    # see genContinInput() for creating storedFn
    storedFn = lines[0][52:].strip()
    infn = sampleDir.parent / storedFn
    if storedFn[0] == '.': # starts with a dot
        infn = sampleDir.parent.parent / (sampleDir.parent.name + storedFn)
    varmap['dataFilename'] = infn
    return dfDistrib, dfFit, varmap