import os 
import sys
import numpy as np
import flopy 
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
import pandas as pd
import scipy.ndimage
import imageio
import re
import csv
import gdal, ogr, os, osr
from pyproj import CRS, Transformer
from osgeo.osr import SpatialReference, CoordinateTransformation
from matplotlib import colors
from matplotlib.colors import LogNorm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
workspace=('C:\\Users\\NLFEGL\\Desktop\\PWN_flopy\\Results\\2D\\')
os.chdir ('C:\\Users\\NLFEGL\\Desktop\\PWN_flopy\\Results') 
modelname='..\\modflowtest.nam'

mf = flopy.seawat.Seawat.load(modelname, exe_name='swt_v4')
swt1= flopy.seawat.Seawat('C:\\Users\\NLFEGL\\Desktop\\PWN_flopy\\Results\\model\\swt1', exe_name='swt_v4', model_ws='C:\\Users\\NLFEGL\\Desktop\\PWN_flopy\\Results\\model')
#workspace=('C:\\Users\\NLFEGL\\Desktop\\PWN_flopy\\Results\\2D\\')
#os.chdir ('C:\\Users\\NLFEGL\\Desktop\\PWN_flopy\\Results\\') 
#modelname='..\\modflowtest.nam'
#mf = flopy.seawat.Seawat.load(modelname, exe_name='swt_v4')
#coordinate transformations(RDnew, EPSG 28992):
lon_1, lon_2= (102000, 106000)
lat_1, lat_2= (507000, 506000)
origin_x, origin_y=(101000, 505000)
ur_x, ur_y= (107000, 510000)
width=50
dist=50
###Translate translation to WGS84(EPSG4326)
# Define the Rijksdriehoek projection system (EPSG 28992)
epsg28992 = SpatialReference()
epsg28992.ImportFromEPSG(28992)
 
# correct the towgs84
epsg28992.SetTOWGS84(565.237,50.0087,465.658,-0.406857,0.350733,-1.87035,4.0812)
 
# Define the wgs84 system (EPSG 4326)
epsg4326 = SpatialReference()
epsg4326.ImportFromEPSG(4326)
rd2latlon = CoordinateTransformation(epsg28992, epsg4326)
latlon2rd = CoordinateTransformation(epsg4326, epsg28992)
latlon1=rd2latlon.TransformPoint(lon_1, lat_1)
latlon2=rd2latlon.TransformPoint(lon_2, lat_2)
rasterLL=rd2latlon.TransformPoint(origin_x, origin_y)
rasterUR=rd2latlon.TransformPoint(ur_x, ur_y)

###Large data formats (DIS, LPF, RCH, VDF)
###Bas
bas_file=(r'modflowtest.bas')
bas_data= pd.read_csv(bas_file, skiprows=3, delim_whitespace=True)
bas_arr= bas_data.to_numpy()
basl1= bas_arr[0:107, :]
basl2= bas_arr[108:216, :]
basl3= bas_arr[217:325, :]
basl4= bas_arr[326:434, :]
basl5= bas_arr[435:543, :]
basl6= bas_arr[544:652, :]
basl7= bas_arr[653:761, :]
basl8= bas_arr[762:870, :]
basl9= bas_arr[871:979, :]
basl10=bas_arr[980:1088, :]
basl11=bas_arr[1089:1197, :]
basl12=bas_arr[1198:1306, :]
basl13=bas_arr[1307:1415, :]
basl14=bas_arr[1416:1524, :]
basl15=bas_arr[1525:1633, :]
strt1= bas_arr[1635:1743, :]
strt2= bas_arr[1744:1852, :]
strt3= bas_arr[1853:1961, :]
strt4= bas_arr[1962:2070, :]
strt5= bas_arr[2071:2179, :]
strt6= bas_arr[2180:2288, :]
strt7= bas_arr[2289:2397, :]
strt8= bas_arr[2398:2506, :]
strt9= bas_arr[2507:2615, :]
strt10=bas_arr[2616:2724, :]
strt11=bas_arr[2725:2833, :]
strt12=bas_arr[2834:2942, :]
strt13=bas_arr[2943:3051, :]
strt14=bas_arr[3052:3160, :]
strt15=bas_arr[3161:3269, :]

###Dis
dis_file=(r'modflowtest.dis')
dis_data= pd.read_csv(dis_file, skiprows=8, delim_whitespace=True)
dis_arr= dis_data.to_numpy()
botml0=dis_arr[0:107, :]
botml1= dis_arr[108:216, :]
botml1=botml1.astype(float)
botml2= dis_arr[217:325, :]
botml3= dis_arr[326:434, :]
botml4= dis_arr[435:543, :]
botml5= dis_arr[544:652, :]
botml6= dis_arr[653:761, :]
botml7= dis_arr[762:870, :]
botml8= dis_arr[871:979, :]
botml9= dis_arr[980:1088, :]
botml10= dis_arr[1089:1197, :]
botml11= dis_arr[1198:1306, :]
botml12= dis_arr[1307:1415, :]
botml13= dis_arr[1416:1524, :]
botml14= dis_arr[1525:1633, :]
botml15= dis_arr[1634:1742, :]

###LPF
lpf_file=(r'modflowtest.lpf')
lpf_data= pd.read_csv(lpf_file, skiprows=8, delim_whitespace=True)
lpf_arr= lpf_data.to_numpy()
hk1= lpf_arr[0:107, :]
vk1= lpf_arr[108:216, :]
hk2= lpf_arr[217:325, :]
vk2= lpf_arr[326:434, :]
hk3= lpf_arr[435:543, :]
vk3= lpf_arr[544:652, :]
hk4= lpf_arr[653:761, :]
vk4= lpf_arr[762:870, :]
hk5= lpf_arr[871:979, :]
vk5= lpf_arr[980:1088, :]
hk6= lpf_arr[1089:1197, :]
vk6= lpf_arr[1198:1306, :]
hk7= lpf_arr[1307:1415, :]
vk7= lpf_arr[1416:1524, :]
hk8= lpf_arr[1525:1633, :]
vk8= lpf_arr[1634:1742, :]
hk9= lpf_arr[1743:1851, :]
vk9= lpf_arr[1852:1960, :]
hk10= lpf_arr[1961:2069, :]
vk10= lpf_arr[2070:2178, :]
hk11= lpf_arr[2179:2287, :]
vk11= lpf_arr[2288:2396, :]
hk12= lpf_arr[2397:2505, :]
vk12= lpf_arr[2506:2614, :]
hk13= lpf_arr[2615:2723, :]
vk13= lpf_arr[2724:2832, :]
hk14= lpf_arr[2833:2941, :]
vk14= lpf_arr[2942:3050, :]
hk15= lpf_arr[3051:3159, :]
vk15= lpf_arr[3160:3268, :]

###RCH
rch_file=(r'modflowtest.rch')
rch_data= pd.read_csv(rch_file, skiprows=4, delim_whitespace=True)
rch_arr= rch_data.to_numpy()
rch1= rch_arr[0:107, :]


###VDF
vdf_file=(r'modflowtest.vdf')
vdf_data= pd.read_csv(vdf_file, skiprows=6, delim_whitespace=True)
vdf_arr= vdf_data.to_numpy()
vdf1= vdf_arr[0:107, :]
vdf2= vdf_arr[108:216, :]
vdf3= vdf_arr[217:325, :]
vdf4= vdf_arr[326:434, :]
vdf5= vdf_arr[435:543, :]
vdf6= vdf_arr[544:652, :]
vdf7= vdf_arr[653:761, :]
vdf8= vdf_arr[762:870, :]
vdf9= vdf_arr[871:979, :]
vdf10= vdf_arr[980:1088, :]
vdf11= vdf_arr[1089:1197, :]
vdf12= vdf_arr[1198:1306, :]
vdf13= vdf_arr[1307:1415, :]
vdf14= vdf_arr[1416:1524, :]
vdf15= vdf_arr[1525:1633, :]


###array2raster
def array2raster(NewRasterfn, rasterOrigin, pixelWidth, pixelHeight, array):
    cols=array.shape[1]
    rows = array.shape[0]
    originX= rasterOrigin[0]
    originY= rasterOrigin[1]
    
    reversed_arr=array[::-1]
    
    driver = gdal.GetDriverByName('GTiff')
    outRaster= driver.Create(NewRasterfn, cols, rows, 1, gdal.GDT_Float64)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband= outRaster.GetRasterBand(1)
    outband.WriteArray(reversed_arr)
    outRasterSRS= osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
#array2raster('2D\\model_top_ras.tiff',(4.5936656, 52.5306043) , ((4.6813581-4.5936656)/128), ((52.5760984-52.5306043)/108), botml0)    



###Make transect
def transect(src, lon_1, lon_2, lat_1, lat_2, width, dist, tifout, csvout):
    ds=gdal.Open(src)
    #print (ds.GetMetadata())
    #coordinate transform 
    proj_str = "+proj=tpeqd +lon_1={} +lat_1={} +lon_2={} +lat_2={}".format(lon_1, lat_1, lon_2, lat_2)
    tpeqd= CRS.from_proj4(proj_str)
    transformer= Transformer.from_crs(CRS.from_proj4("+proj=latlon"), tpeqd)
    
    #transfer to tpeqd coordinates
    point_1= transformer.transform(lon_1, lat_1)
    point_2= transformer.transform(lon_2, lat_2)
    print (point_1)
    print (point_2)
    #create box in tpeqd coordinates
    bbox = (point_1[0], -(width*0.5), point_2[0], (width*0.5))
    
    #calculate number of samples
    num_samples = int((point_2[0] - point_1[0]) / dist)
    #num_samples = 100
    #Warp it into dataset in tpeqd projection
    format='GTiff'
    profile= gdal.Warp(tifout, ds, dstSRS=proj_str, outputBounds=bbox, height=1, width=num_samples, resampleAlg='near', format=format)
    
    #Extract pixel values and write output file
    data = profile.GetRasterBand(1).ReadAsArray()
    
    #Write csv output
    with open(csvout, 'w') as f:
        f.write("dist,value\n")
        for (d, value) in enumerate(data[0, :]):
            f.write("{}, {}\n".format(d*dist, value))
        print("saves as{}".format(csvout))
    
    #Clean up
    #profile= None
    #ds= None

#transect('model_top_ras.tiff', 4.6080825, 4.667197, 52.5486755, 52.5400629, 10, 10, 'tifout.tif', 'line_values.csv')

###csv to array
def csv2array(csv_file):
    read=r'C:\\Users\\NLFEGL\\Desktop\\PWN_flopy\\Results\\2D\\'+csv_file
    read_data=pd.read_csv(read, skiprows=1)
    read_data.columns= [ 'dist', 'values']
    read_data=read_data.loc[:, [ 'dist', 'values']]
    read_data=np.asarray(read_data['values'])
    return(read_data)


###Variable arrays
baslayer=[basl1, basl2, basl3, basl4, basl5, basl6, basl7, basl8, basl9, basl10, basl11, basl12, basl13, basl14, basl15]
botmlayer=[botml1, botml2, botml3, botml4, botml5, botml6, botml7, botml8, botml9, botml10, botml11, botml12, botml13, botml14, botml15]
strtlayer=[strt1, strt2, strt3, strt4, strt5, strt6, strt7, strt8, strt9, strt10, strt11, strt12, strt13, strt14, strt15]
hklayer=[hk1, hk2, hk3, hk4, hk5, hk6, hk7, hk8, hk9, hk10, hk11, hk12, hk13, hk14, hk15]
vklayer=[vk1, vk2, vk3, vk4, vk5, vk6, vk7, vk8, vk9, vk10, vk11, vk12, vk13, vk14, vk15]
vdflayer=[vdf1, vdf2, vdf3, vdf4, vdf5, vdf6, vdf7, vdf8, vdf9, vdf10, vdf11, vdf12, vdf13, vdf14, vdf15]
rchlayer=[rch1]
variables=[baslayer, botmlayer, strtlayer, hklayer, vklayer, vdflayer, rchlayer]
str_variables=['basl', 'botml', 'strt', 'hk', 'vk', 'vdf', 'rch']
###Producing input values
#model_top
array2raster(workspace+'botml0.tiff',(rasterLL[1], rasterLL[0]) , ((rasterUR[1]-rasterLL[1])/128), ((rasterUR[0]-rasterLL[0])/108), botml0)
transect(workspace+'botml0.tiff', latlon1[1], latlon2[1], latlon1[0], latlon2[0], width, dist, workspace+'trans_botml0.tiff', workspace+'botml0_values.csv')
top_array=csv2array('botml0_values.csv')
#other inputs
for x in range(len(variables)):
    variable=variables[x]
    for y in range(len(variable)):
        a=workspace+str_variables[x]+str(y+1)
        array2raster(a+'.tiff',(rasterLL[1], rasterLL[0]) , ((rasterUR[1]-rasterLL[1])/128), ((rasterUR[0]-rasterLL[0])/108), variable[y])
        transect(workspace+str_variables[x]+str(y+1)+'.tiff', latlon1[1], latlon2[1], latlon1[0], latlon2[0], width, dist, workspace+'trans_'+str_variables[x]+str(y+1)+'.tiff', workspace+str_variables[x]+str(y+1)+'_values.csv') 
        variable[y]=csv2array(str_variables[x]+str(y+1)+'_values.csv')

###Spatial discretization
nlay = 15
nrow = len(hklayer[0])
ncol = len(hklayer[0])
delr=1
delc=1
perlen=86400
ntst=15
###Transforming 2D-Arrays to quasi-3D arrays
basl3d=np.ones((nlay, len(hklayer[0]), len(hklayer[0])))
topl3d=np.ones((len(rchlayer), len(hklayer[0]), len(hklayer[0])))
botml3d=np.ones((nlay, len(hklayer[0]), len(hklayer[0])))
strt3d=np.ones((nlay, len(hklayer[0]), len(hklayer[0])))
hk3d=np.ones((nlay, len(hklayer[0]), len(hklayer[0])))
vk3d=np.ones((nlay, len(hklayer[0]), len(hklayer[0])))
vdf3d=np.ones((nlay, len(hklayer[0]), len(hklayer[0])))
rch3d=np.ones((len(rchlayer), len(hklayer[0]), len(hklayer[0])))
for h in range(len(hklayer)):
    for g in range(len(hklayer[0])):
        basl3d[h, g, :]=baslayer[h]
for h in range(len(hklayer)):
    for g in range(len(hklayer[0])):
        botml3d[h, g, :]=botmlayer[h]
for h in range(len(hklayer)):
    for g in range(len(hklayer[0])):
        strt3d[h, g, :]=strtlayer[h]
for h in range(len(hklayer)):
    for g in range(len(hklayer[0])):
        hk3d[h, g, :]=hklayer[h]
for h in range(len(hklayer)):
    for g in range(len(hklayer[0])):
        vk3d[h, g, :]=vklayer[h]
for h in range(len(hklayer)):
    for g in range(len(hklayer[0])):
        vdf3d[h, g, :]=vdflayer[h]
for h in range(len(rchlayer)):
    for g in range(len(hklayer[0])):
        rch3d[h, g, :]=rchlayer[h]
for h in range(len(rchlayer)):
    for g in range(len(hklayer[0])):
        topl3d[h, g, :]=top_array
print(topl3d.shape)


###Modflow package building
dis=flopy.modflow.ModflowDis(swt1, nlay, nrow, ncol, delr=delr, delc=delc, top=topl3d, botm=botml3d, nper=1, perlen=36500, nstp=36500, steady=True, itmuni=4)
#swt1.dis.plot()
#print(swt1.namefile)
#Create Bas package
basl3d[:,:, 0 ]= -1
bas=flopy.modflow.mfbas.ModflowBas(swt1,ibound=basl3d, strt=strt3d)

#LFP package
lpf=flopy.modflow.ModflowLpf(swt1, hk=hk3d, vka=vk3d, ipakcb=53)
#swt1.lpf.plot()
#plot hk, vk
#hor_cond=swt1.lpf.hk.array
#print(hor_cond[:, 0, :])
fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1)
xsect = flopy.plot.PlotCrossSection(model=swt1, line={'Row': 0})
#csa = xsect.plot_array()
patches = xsect.plot_ibound()
linecollection = xsect.plot_grid(linewidth=0.2)
t = ax.set_title('Cross-Section with Horizontal hydraulic conductivities')

#N = 100
#X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
#pcm = ax.pcolor(X, Y, hor_cond, norm=colors.LogNorm(vmin=hor_cond.min(), vmax=hor_cond.max()))
#cb = plt.colorbar(csa, shrink=0.75)
#PCG package
pcg=flopy.modflow.ModflowPcg(swt1)

#Output Control
oc=flopy.modflow.ModflowOc(swt1, stress_period_data={(0, 0): ['save head', 'save budget']}, compact=True, save_every=True)

#Variable density flow
#vdf=flopy.seawat.SeawatVdf(swt1, vdf=vdf3d)

#RCH
rch=flopy.modflow.ModflowRch(swt1, rech=rch3d)

###Run model
swt1.write_input()
swt1.run_model()

#Extract heads
fname='C:\\Users\\NLFEGL\\Desktop\\PWN_flopy\\Results\\model\\swt1.hds'
headobj=bf.HeadFile(fname)
times=headobj.get_times()
head=headobj.get_data()
head[0, :, 40, :]
print (head.shape)

levels= MaxNLocator(nbins=15).tick_values(head.min(), head.max())
cmap=plt.get_cmap('Blues')
norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True)
fig=plt.figure(figsize=(20, 10))
ax=fig.add_subplot(1, 1, 1, aspect='equal')
im=ax.imshow(head[0, :, 40, :], interpolation='nearest', extent=(0, 81, 0, 15))
ax.set_title('heads at time=0')

#os.chdir('C:\\Users\\NLFEGL\\Desktop\\PWN_flopy\\Results\\model')
#fname = 'swt1.hds'
#headobj = bf.HeadFile(fname)
#times = headobj.get_times()
#head = headobj.get_data(totim=times[-1])
##levels = MaxNLocator(nbins=15).tick_values(head.min(), head.max())
#cmap = plt.get_cmap('Blues')
##norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#fig = plt.figure(figsize=(20, 10))
#ax = fig.add_subplot(1, 1, 1, aspect='equal')
#im = ax.imshow(head[:, 0, :])

#ax.set_title('Simulated Heads')
#plt.savefig('HeadDistribution.png')

#fig = plt.figure(figsize=(18, 5))
#ax = fig.add_subplot(1, 1, 1)
#xsect = flopy.plot.PlotCrossSection(model=swt1, line={'Row': 0})
#csa = xsect.plot_array(head)

#contour_set = xsect.contour_array(head, cmap='jet')
#patches = xsect.plot_ibound()
#linecollection = xsect.plot_grid(linewidth=0.1)
#t = ax.set_title('Cross-Section with Head values')
##fig.colorbar(im, ax=ax, fraction=0.05, label="Head (m)")
#water_table=flopy.utils.postprocessing.get_water_table(head, nodata=-9999)
#print(water_table)
##water_table=water_table.reshape(1, 410)
#im=ax.plot(np.arange(410), water_table, 'ro')
##xsect.plot_array(water_table)
#print(water_table)

