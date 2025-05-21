'''

Description: This script is a simple way to read in radar files and perform a so-called
four quadrant analysis summarized in Alford et al. (2025, GRL). DOI: XXXXX.

Author: Addison Alford

Running the script: python 4quadDecompFinal.py meso_input_file.txt

Meso input file: Comma delimited file with the radar file path, the meso range from radar (km), the meso azimuth from north (deg)

'''

import sys
import pyart
import numpy as np
from datetime import datetime
import matplotlib.dates as dttime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from netCDF4 import Dataset


def interpVar(var,varMask,spatial):

    '''
    +++ Description +++

    This function takes the velocity field input (vel) and returns a smooted
    velocity field and the difference of the raw field from the smoothed.

    +++ Input +++

    vel:        The observed Doppler velocity field.

    spatial:    Either the radius or azimuth data associated with the shape of vel.

    filter:     The radial or azimuthal filter window.

    +++ Returns +++

    The smoothed velocity field and the difference from the observed velocity field.
    
    '''

    #Linear interpolation along radial.
    velInterp = np.empty(var.shape)
    for ray,rayMask,num in zip(var,varMask,list(range(velInterp.shape[0]))):
        if ray[rayMask == False].shape[0] > 5:
            spatialIn = spatial[rayMask == False]
            rayIn = ray[rayMask == False]
            indices = spatialIn.argsort()
            velInterp[num] = np.interp(spatial,spatialIn[indices],rayIn[indices])
        else: velInterp[num] = ray

    return velInterp

def savgolFilt(varInterp,filter):

    #Smooth interpolated field.
    velSmooth = savgol_filter(varInterp, filter, 3, mode='mirror',axis=1)

    #Change -32768 to NaNs.
    velSmooth[velSmooth < -1000] = np.nan

    return velSmooth

#Simple estimate of KDP using the Savgol Filter, which is a running, linear-least squares fit to data.   
def kdpComputation(phiDP,radial,rspace = 250):

    kdp = np.zeros(phiDP.shape)

    window = int(6000 / rspace)

    if window%2 == 0:
        window+=1

    phiInterp = interpVar(phiDP,phiDP.mask,radial)

    phiSmooth = savgolFilt(phiInterp,window) #LLSQ Fit to PHIDP

    kdp[:,1::] = np.diff(phiSmooth,axis=1) / (2. * rspace)

    kdp = np.ma.masked_where(phiDP.mask,kdp)
    
    return kdp*1000 #deg/km

def fourPanel(fList,rList=0,azList=0,samList=None,zmax=10,kdpComp = False,qc=False,mesoName='meso',tornadoTime=None):

    """
    Input

    fname:  The name of the cfradial file.

    rMeso:  The radius from the radar to the mesovortex location in km.

    azMeso:  The azimuth for to the mesovortex location in degrees.

    zmax:   The top of the cross section in km.

    kdpComp:    Compute KDP?

    Ouput

    Creates RHI plot.

    """

    refSeries,zdrSeries,kdpSeries,timeInfo,elInfo,heightSeries=[],[],[],[],[],[]

    #The way I set this up was to input the cfrad files in a list with r, theta associated with each file.
    #For each 2nd 0.5 cut (SAILS cut), I spoofed the filename to have a start time associated with the SAILS cut 
    #and then record the position of the mesovortex in the 2nd cut.

    for fname,rMeso,azMeso,mesoNum in zip(fList[::2],rList[::2],azList[::2],range(len(fList))[::2]):

        mesoDir = np.degrees(np.arctan2(rList[mesoNum+1]*np.sin(np.radians(azList[mesoNum+1])) - rList[mesoNum]*np.sin(np.radians(azList[mesoNum])),\
            rList[mesoNum+1]*np.cos(np.radians(azList[mesoNum+1])) - rList[mesoNum]*np.cos(np.radians(azList[mesoNum]))))

        startTime,endTime = fList[mesoNum].split('/')[-1].split('.')[1],fList[mesoNum+1].split('/')[-1].split('.')[1]
        
        deltaT = (datetime.strptime(endTime,"%Y%m%d_%H%M%S")-datetime.strptime(startTime,"%Y%m%d_%H%M%S")).seconds

        uMeso,vMeso = 1000*(rList[mesoNum+1]*np.sin(np.radians(azList[mesoNum+1])) - rList[mesoNum]*np.sin(np.radians(azList[mesoNum])))/deltaT,\
            1000*(rList[mesoNum+1]*np.cos(np.radians(azList[mesoNum+1])) - rList[mesoNum]*np.cos(np.radians(azList[mesoNum])))/deltaT
        
        radar = pyart.io.read(fname)

        x,y,z = radar.gate_x['data'][:]/1000,radar.gate_y['data'][:]/1000,radar.gate_z['data'][:]/1000

        r = radar.range['data'][:]

        ref,vel,zdr,rho,phiDP = radar.fields['REF']['data'][:].copy(),radar.fields['VEL']['data'][:].copy(),\
            radar.fields['ZDR']['data'][:].copy(),radar.fields['RHO']['data'][:].copy(),\
            radar.fields['PHI']['data'][:].copy()

        if qc:
            mask = (((zdr>8)&(rho<0.8)) | (rho > 1.00) | (zdr>10) | (rho < 0.6))
            ref,vel,rho,phiDP,zdr = np.ma.masked_where(mask,ref),np.ma.masked_where(mask,vel),\
                np.ma.masked_where(mask,rho),np.ma.masked_where(mask,phiDP),np.ma.masked_where(mask,zdr)
        
        radar.fields['REF']['data'][:],radar.fields['VEL']['data'][:],\
            radar.fields['ZDR']['data'][:],radar.fields['RHO']['data'][:],\
            radar.fields['PHI']['data'][:] = ref.copy(),vel.copy(),zdr.copy(),rho.copy(),phiDP.copy()

        if kdpComp:
            if 'KDP' not in list(radar.fields.keys()):
                kdp = kdpComputation(phiDP,r,rspace = r[1]-r[0])
                if 'KDP' not in list(radar.fields.keys()):
                    radar.add_field('KDP',\
                                    radar.fields['PHI'].copy())
                    radar.fields['KDP']['data'][:] = kdp.copy()
            else:
                kdp = radar.fields['KDP']['data'][:].copy()
        
        xMeso,yMeso = rMeso*np.sin(np.radians(azMeso)),rMeso*np.cos(np.radians(azMeso))

        timeGrid,_ = np.meshgrid(radar.time['data'],radar.range['data'],indexing='ij')

        xMesoRel,yMesoRel = x - (xMeso + 0.001*uMeso*timeGrid),y - (yMeso + 0.001*vMeso*timeGrid)

        rMesoRel,dirMesoRel = np.sqrt(xMesoRel**2 + yMesoRel**2),np.degrees(np.arctan2(xMesoRel,yMesoRel))
        
        dirMesoRel+=-mesoDir
        
        dirMesoRel[dirMesoRel<0] = dirMesoRel[dirMesoRel<0] + 360

        dirMesoRel[dirMesoRel>=360] = dirMesoRel[dirMesoRel>=360] - 360
        
        time = fname.split('/')[-1].split('.')[1]


        vars = [{"fName":"reflectivity","name":"Reflectivity","data":ref,"range":range(0,65,5),"cbar":pyart.graph.cm_colorblind.ChaseSpectral,"units":"dBZ"},
                {"fName":"differential_reflectivity","name":"Differential Reflectivity","data":zdr,"range":range(-1,6,1),"cbar":pyart.graph.cm_colorblind.ChaseSpectral,"units":"dB"},
                {"fName":"KDP","name":"Spec. Differential Phase","data":kdp,"range":range(-5,6,1),"cbar":pyart.graph.cm_colorblind.ChaseSpectral,"units":"deg/km"}
        ]

        rMax = 5.

        for var in vars:

            varFourQuad,varFourQuadStd,heightFourQuad = np.empty((radar.nsweeps-1,4)),np.empty((radar.nsweeps-1,4)),np.empty((radar.nsweeps-1,4))
            
            halfFound,afterSails = False,False
            for slice,i in zip(radar.iter_slice(),range(radar.nsweeps)):
                if (halfFound) and (np.abs(np.nanmean(radar.elevation['data'][slice])-0.5) < 0.1):
                    afterSails = True
                else:
                    if halfFound and afterSails: 
                        i = i-1
                    for startAz,j in zip(range(0,360,90),range(4)):
                        
                        indices = ((rMesoRel[slice] < rMax) & (dirMesoRel[slice] >= startAz) &(dirMesoRel[slice] < startAz+90))

                        heightFourQuad[i,j] = np.nanmean(z[slice][indices])
                        try: 
                            varFourQuad[i,j] = np.nanmean(var["data"][slice][indices])
                            varFourQuadStd[i,j] = np.nanstd(var["data"][slice][indices])
                        except: 
                            varFourQuad[i,j] = np.nan
                            varFourQuadStd[i,j] = np.nan
                    if np.abs(np.nanmean(radar.elevation['data'][slice])-0.5) < 0.1:
                        halfFound=True
            if var["fName"] == 'reflectivity':
                refSeries.append(varFourQuad)
            elif var["fName"] == 'differential_reflectivity':
                zdrSeries.append(varFourQuad)
            elif var["fName"] == 'KDP':
                kdpSeries.append(varFourQuad)
            else:
                print("can't find variables")
        heightSeries.append(heightFourQuad)
        timeInfo.append(dttime.date2num(datetime.strptime(time,"%Y%m%d_%H%M%S")))
        elInfo.append(np.unique(radar.fixed_angle["data"][:]))    

    vars = [{"fName":"reflectivity","name":"Reflectivity","data":refSeries,"range":range(0,65,5),"cbar":pyart.graph.cm_colorblind.ChaseSpectral,"units":"dBZ"},
            {"fName":"differential_reflectivity","name":"Differential Reflectivity","data":zdrSeries,"range":range(-1,6,1),"cbar":pyart.graph.cm_colorblind.ChaseSpectral,"units":"dB"},
            {"fName":"KDP","name":"Spec. Differential Phase","data":kdpSeries,"range":range(-5,6,1),"cbar":pyart.graph.cm_colorblind.ChaseSpectral,"units":"deg/km"}
            ]
    
    for var in vars:
        fig = plt.figure(1)
        plt.clf()
        fig.set_size_inches(10,10)

        ax1,ax2,ax3,ax4 = fig.add_subplot(221),fig.add_subplot(222),fig.add_subplot(223),fig.add_subplot(224)
        heightIn = heightSeries
        for varTime,height,el,t in zip(var["data"],heightIn,elInfo,timeInfo):
            varTime,height = varTime.T,height.T
            tGrid=t*np.ones(varTime[0].shape)

            indexing = np.argsort(height[0])
            a=ax1.pcolormesh(np.array([(tGrid-0.0005).tolist(),tGrid.tolist(),(tGrid+0.0005).tolist()]),\
                np.array([height[3].tolist(),height[3].tolist(),height[3].tolist()])[:,indexing],\
                    np.array([varTime[3].tolist(),varTime[3].tolist(),varTime[3].tolist()])[:,indexing],\
                        vmin=np.nanmin(var["range"]),vmax=np.nanmax(var["range"]),cmap=var["cbar"])
            ax2.pcolormesh(np.array([(tGrid-0.0005).tolist(),tGrid.tolist(),(tGrid+0.0005).tolist()]),\
                np.array([height[0].tolist(),height[0].tolist(),height[0].tolist()])[:,indexing],\
                    np.array([varTime[0].tolist(),varTime[0].tolist(),varTime[0].tolist()])[:,indexing],\
                        vmin=np.nanmin(var["range"]),vmax=np.nanmax(var["range"]),cmap=var["cbar"])
            ax4.pcolormesh(np.array([(tGrid-0.0005).tolist(),tGrid.tolist(),(tGrid+0.0005).tolist()]),\
                np.array([height[1].tolist(),height[1].tolist(),height[1].tolist()])[:,indexing],\
                    np.array([varTime[1].tolist(),varTime[1].tolist(),varTime[1].tolist()])[:,indexing],\
                        vmin=np.nanmin(var["range"]),vmax=np.nanmax(var["range"]),cmap=var["cbar"])            
            ax3.pcolormesh(np.array([(tGrid-0.0005).tolist(),tGrid.tolist(),(tGrid+0.0005).tolist()]),\
                np.array([height[2].tolist(),height[2].tolist(),height[2].tolist()])[:,indexing],\
                    np.array([varTime[2].tolist(),varTime[2].tolist(),varTime[2].tolist()])[:,indexing],\
                        vmin=np.nanmin(var["range"]),vmax=np.nanmax(var["range"]),cmap=var["cbar"])        

        if tornadoTime != None:

            tornadoNum = dttime.date2num(datetime.strptime(tornadoTime,"%Y%m%d_%H%M"))

            for ax in [ax1,ax2,ax3,ax4]:
                ax.plot([tornadoNum,tornadoNum],[0,10],'k--')

        ax1.xaxis.set_ticks(timeInfo)
        ax1.xaxis.set_ticklabels([])
        ax2.xaxis.set_ticks(timeInfo)
        ax2.xaxis.set_ticklabels([])
        ax3.xaxis.set_ticks(timeInfo)
        ax3.xaxis.set_ticklabels([datetime.strftime(dttime.num2date(t),"%H%M%S") for t in timeInfo],rotation=45,weight='bold',size=15)
        ax4.xaxis.set_ticks(timeInfo)
        ax4.xaxis.set_ticklabels([datetime.strftime(dttime.num2date(t),"%H%M%S") for t in timeInfo],rotation=45,weight='bold',size=15)

        yticks = range(0,11,2)
        ax1.yaxis.set_ticks(yticks)
        ax1.yaxis.set_ticklabels(yticks,weight='bold',size=15)
        ax2.yaxis.set_ticks(yticks)
        ax2.yaxis.set_ticklabels([])
        ax3.yaxis.set_ticks(yticks)
        ax3.yaxis.set_ticklabels(yticks,weight='bold',size=15)
        ax4.yaxis.set_ticks(yticks)
        ax4.yaxis.set_ticklabels([])

        for ax in [ax1,ax2,ax3,ax4]:
            ax.set_ylim(0,6)

        fig.subplots_adjust(left=0.05,bottom=0.22,top=0.98,right=0.95,wspace=0.05,hspace=0.05)

        cax = [0.1,0.1,0.8,0.03]
        cb = plt.colorbar(a,orientation='horizontal',cax = fig.add_axes(cax))
        cb.set_ticks(var["range"])
        cb.set_label(var["units"])
        cb.ax.set_xticklabels(var["range"])

        fig.savefig("fourquad_%s_%s.png"%(var["fName"],mesoName),dpi=300)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        raise FileNotFoundError("You must specificy a mesovortex file with the radar fileName, r, theta as a csv delimited row.")

    inFile = np.genfromtxt(sys.argv[1],delimiter=',',dtype=str).T

    mesoName = sys.argv[1].split('/')[-1].split('.')[0]

    fList,rList,azList = [f for f in inFile[0]],[float(r) for r in inFile[1]],[float(az) for az in inFile[2]]
    
    #Tornado times identifed by the NWS in UTC.
    if 'gracemont' in sys.argv[1]: tornadoTime = "20230227_0232"
    if 'minco' in sys.argv[1]: tornadoTime = "20230227_0252"
    if 'tuttle' in sys.argv[1]: tornadoTime = "20230227_0257"

    fourPanel(fList,rList=rList,azList=azList,zmax=15,kdpComp=True,qc=True,mesoName=mesoName,tornadoTime = tornadoTime)

    
