import sys
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pyart
import plotly.graph_objects as go

'''

Description: This script is a simple way to read in a radar file and a SAMURAI analysis and plot
them in 3D using the Plotly library.

Author: Addison Alford

Running the script: python plot3D.py meso_input_file.txt

Meso input file: Comma delimited file with the radar file path, the meso range from radar (km), the meso azimuth from north (deg), and the SAMURAI file path

'''

### Grid specifications from SAMURAI analysis ###

xmin,xmax,dx = -110,10,0.5
ymin,ymax,dy = -60,40,0.5
zmin,zmax,dz = 0,10,0.5

#Must have an input file (see comments above)
inFile = np.genfromtxt(sys.argv[1],delimiter=',',dtype=str).T

#Extract info from input file
fList,rList,azList,samList = [f for f in inFile[0]],[float(r) for r in inFile[1]],[float(az) for az in inFile[2]],[f for f in inFile[3]]

mesoName = sys.argv[1].split('/')[-1].split('.')[0]

#Note that in my input file, there are "ghost" radar files on every other line just to denote the time of the 2nd 0.5 scan (that is not used in the analysis) but is 
#used to track the low level mesovortex location. Hence, the [::2].
for radarFile,mesoRange,mesoAz,samFile in zip(fList[::2],rList[::2],azList[::2],samList[::2]):
    
    #Read in some aspects of the SAMURAI file
    time = samFile.split("/")[-3]

    nc = Dataset(samFile)

    limit = 5 #km

    x,y,z = nc.variables['x'][:],nc.variables['y'][:],nc.variables['altitude'][:]

    u,v = nc.variables['U'][0].T,nc.variables['V'][0].T

    dudx,dudy,_ = np.gradient(u,x*1000,y*1000,z)

    dvdx,dvdy,_ = np.gradient(v,x*1000,y*1000,z)

    #Read in the ATD file
    radar = pyart.io.read(radarFile)

    #Interp the ATD KDP data to same Cartesian grid used in SAMURAI
    gridInterp=pyart.map.grid_from_radars(
                radar, grid_shape=(z.shape[0],y.shape[0],x.shape[0]), grid_limits=((zmin*1e3,zmax*1e3),(ymin*1e3,ymax*1e3),(xmin*1e3,xmax*1e3)),\
                gridding_algo='map_gates_to_grid',grid_origin=(radar.latitude['data'][0],radar.longitude['data'][0]),
                grid_origin_alt=None, grid_projection=None,
                fields=["KDP","AZ"], gatefilters=None, map_roi=True,
                weighting_function='Barnes2', toa=17000.0, roi_func='constant',constant_roi=1000,
                min_radius=1000.0,
                h_factor=0.1, nb=3.0, bsp=1.0)
    
    #Extract KDP
    kdp = gridInterp.fields['KDP']['data']

    #Create 3D grid
    xgrid,ygrid,zgrid=np.meshgrid(x,y,z,indexing='ij')

    #Compute meso location
    xMeso,yMeso = mesoRange*np.sin(np.radians(mesoAz)),mesoRange*np.cos(np.radians(mesoAz))
    
    #Create and mask the divergence field
    div = np.ma.masked_where(((xgrid<xMeso-2*limit)|(xgrid>xMeso+2*limit)|(ygrid<yMeso-2*limit)|(ygrid>yMeso+2*limit)),dudx+dvdy)

    #Extract and mask the w field
    wnew = np.ma.masked_where(((xgrid<xMeso-limit)|(xgrid>xMeso+limit)|(ygrid<yMeso-limit)|(ygrid>yMeso+limit)),nc.variables["W"][0].T).filled(fill_value=-999.)

    #Extract and mask the KDP field
    kdp = np.ma.masked_where(((xgrid<xMeso-limit)|(xgrid>xMeso+limit)|(ygrid<yMeso-limit)|(ygrid>yMeso+limit)),kdp.T).filled(fill_value=np.nan)
    
    xgrid,ygrid,zgrid = np.mgrid[x.min():x.max()+dx:dx,y.min():y.max()+dy:dy,\
        z.min():z.max()+dz:dz]
    
    #Top level = 2km altitude
    topLev = 5
    
    #Plotly code
    lower,upper = -10,10
    mask = np.ma.masked_outside(wnew,lower,upper).mask
    xgrid = np.ma.masked_where((mask&(zgrid>=topLev)),xgrid)
    ygrid = np.ma.masked_where((mask&(zgrid>=topLev)),ygrid)
    zgrid = np.ma.masked_where((mask&(zgrid>=topLev)),zgrid)
    wnew = np.ma.masked_where((mask&(zgrid>=topLev)),wnew)    
    
    #Limit from meso center for divergence field
    limit = 7.5
    
    #Divergence field projected onto z=0km
    fig=go.Figure(data=go.Surface(x=xgrid[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,0],
        y=ygrid[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,0],
        z=zgrid[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,0]+div[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,1],
        surfacecolor=zgrid[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,0]+div[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,1],
        cmin=zgrid[0,0,0]-0.010,cmax=zgrid[0,0,0]+0.010,showscale=True,opacity=0.9,colorscale='RdBu',
        contours=dict(z=dict(start=zgrid[0,0,0]-.010,end=zgrid[0,0,0]+0.010,usecolormap=True,size=.001,show=True,project_z=False)
            )))
    #Add the meso center  
    fig.add_trace(go.Scatter3d(x=[xMeso],y=[yMeso],z=[zgrid[0,0,0]],mode='markers',marker=dict(color='black',size=3),showlegend=False))

    #Limit from meso center for W and KDP field
    limit=5

    for i in range(1,topLev):
        #Plot for the vertical velocity
        fig.add_trace(go.Surface(x=xgrid[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,0],
            y=ygrid[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,0],
            z=zgrid[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,i],
            surfacecolor=wnew[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,i],
            cmin=lower,cmax=upper,cmid=0,showscale=False,opacity=0.9,colorscale='icefire'))
        
        #Contours for KDP.
        fig.add_trace(go.Surface(x=xgrid[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,0],
            y=ygrid[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,0],
            z=zgrid[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,i]+kdp[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,i]/1000,
            surfacecolor=zgrid[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,i]+kdp[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,i]/1000,
            cmin=zgrid[0,0,i]+.001,cmax=zgrid[0,0,i]+0.005,opacity=0,showscale=False,colorscale='Greys',
            contours=dict(z=dict(start=zgrid[0,0,i]+.001,end=zgrid[0,0,i]+0.005,color='black',size=.001,show=True,project_z=False,width=16)
            )))
        
        #print(time,i,np.nanmax(kdp[((x>=xMeso-limit)&(x<=xMeso+limit))][:,((y>=yMeso-limit)&(y<=yMeso+limit)),:][:,:,i]))

        #Add the meso center
        fig.add_trace(go.Scatter3d(x=[xMeso],y=[yMeso],z=[zgrid[0,0,i]],mode='markers',marker=dict(color='black',size=3),showlegend=False))

    #Set the "camera" angle
    camera = dict(eye=dict(x=1.25,y=-1.25,z=.75))
    
    scene = dict(zaxis=dict(showticklabels=False))

    fig.update_layout(scene_camera=camera,scene=scene,title=time+' UTC',font=dict(size=13,variant="small-caps"))

    fig.write_image("%s_%s.svg"%(mesoName,time),width=500,height=500,scale=3)

