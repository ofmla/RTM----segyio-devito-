import numpy as np
import segyio, sys
import time, math
import cloudpickle as pickle
#from dask_jobqueue import SLURMCluster
from distributed import Client, LocalCluster, wait, as_completed, get_worker

from examples.seismic import Model,AcquisitionGeometry
from examples.seismic.tti import AnisotropicWaveSolver
from skimage.transform import rescale, resize, downscale_local_mean
from devito import *
from examples.seismic import PointSource
from examples.seismic.tti.operators import kernel_centered_2d, Gxx_centered_2d,Gzz_centered_2d

def humanbytes(B):
   'Return the given bytes as a human friendly KB, MB, GB, or TB string'
   B = float(B)
   KB = float(1024)
   MB = float(KB ** 2) # 1,048,576
   GB = float(KB ** 3) # 1,073,741,824
   TB = float(KB ** 4) # 1,099,511,627,776
   
   if B < KB:
      return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
   elif KB <= B < MB:
      return '{0:.2f} KB'.format(B/KB)
   elif MB <= B < GB:
      return '{0:.2f} MB'.format(B/MB)
   elif GB <= B < TB:
      return '{0:.2f} GB'.format(B/GB)
   elif TB <= B:
      return '{0:.2f} TB'.format(B/TB)
      
def remove_out_of_bounds_receivers(origin, rec_coord):
    'Only keep receivers within the model'
    xmin = origin[0]
    idx_xrec = np.where(rec_coord[:,0] < xmin)[0]
    is_empty = idx_xrec.size == 0
    if not is_empty:
       rec_coord=np.delete(rec_coord, idx_xrec, axis=0)

    # For 3D shot records, scan also y-receivers
    if len(origin) == 3:
        ymin = origin[1]
        idx_yrec = np.where(rec_coord[:,1] < ymin)[0]
        is_empty = idx_yrec.size == 0
        if not is_empty:
           rec_coord=np.delete(rec_coord, idx_yrec, axis=0)
           
def limit_model_to_receiver_area(rec_coord,src_coord,origin,spacing,shape,vel,par,
                                 space_order=16, nbl=40, buffer=0.):
    'Restrict full velocity model to area that contains either sources and receivers'
    ndim = len(origin)
    
    # scan for minimum and maximum x and y source/receiver coordinates
    min_x = min(src_coord[0],np.amin(rec_coord[:,0]))
    max_x = max(src_coord[0],np.amax(rec_coord[:,0]))
    if ndim == 3:
        min_y = min(src_coord[1],np.amin(rec_coord[:,1]))
        max_y = max(src_coord[0],np.amax(rec_coord[:,0]))
        
    # add buffer zone if possible
    min_x = max(origin[0], min_x-buffer)
    max_x = min(origin[0] + spacing[0]*(shape[0]-1), max_x+buffer)
    if ndim == 3:
        min_y = max(origin[1], min_y-buffer)
        max_y = min(origin[1] + spacing[1]*(shape[1]-1), max_y+buffer)
        
    # extract part of the model that contains sources/receivers
    nx_min = int(round((min_x - origin[0])/spacing[0])) + 1
    nx_max = int(round((max_x - origin[0])/spacing[0])) + 1
    if ndim == 2:
        ox = float((nx_min - 1)*spacing[0])
        oz = origin[-1]
    else:
        ny_min = int(round(min_y/spacing[1])) + 1
        ny_max = int(round(max_y/spacing[1])) + 1
        ox = float((nx_min - 1)*spacing[0])
        oy = float((ny_min - 1)*spacing[1])
        oz = origin[-1]
        
    # Extract relevant model part from full domain
    if ndim == 2:
        vel = vel[nx_min: nx_max, :]
        delta= par[0][nx_min: nx_max, :]
        epsilon= par[1][nx_min: nx_max, :]
        theta= par[2][nx_min: nx_max, :]
        phi=None
        origin = (ox, oz)
    else:
        vel = vel[nx_min:nx_max,ny_min:ny_max,:]
        delta= par[0][nx_min:nx_max,ny_min:ny_max,:]
        epsilon= par[1][nx_min:nx_max,ny_min:ny_max,:]
        theta= par[2][nx_min:nx_max,ny_min:ny_max,:]
        phi= par[3][nx_min:nx_max,ny_min:ny_max,:]
        origin = (ox, oy, oz)

    return Model(vp=vel, origin=origin, shape=vel.shape, spacing=spacing,
              space_order=space_order, nbl=nbl, epsilon=epsilon, 
               delta=delta, theta=theta, phi=phi, dtype=np.float32)

def main(c):
    par_dir_files='ModelParams/'
    shot_dir_files='ModelShots/'
    par_files=['Delta_Model.sgy','Epsilon_Model.sgy','Theta_Model.sgy','Vp_Model.sgy']
    shot_files=['Anisotropic_FD_Model_Shots_part1.sgy', 'Anisotropic_FD_Model_Shots_part2.sgy', 
            'Anisotropic_FD_Model_Shots_part3.sgy','Anisotropic_FD_Model_Shots_part4.sgy']
    par_files = [par_dir_files + sub for sub in par_files] 
    shot_files = [shot_dir_files + sub for sub in shot_files]

    # Read shots
    start = time.time()
    f = segyio.open(shot_files[0], ignore_geometry=True)
    num_samples = len(f.samples) # number of samples
    samp_int = f.bin[segyio.BinField.Interval]/1000 # samples interval
    records=f.group(segyio.TraceField.FieldRecord) # groups with the same FieldRecord
    print(next(records.values()))
    print("Number of samples {}".format(num_samples))
    print("Sampling interval {}".format(samp_int))

    if int(f.header[0][segyio.TraceField.ElevationScalar]) != 1: 
      scalel=abs(1./f.header[0][segyio.TraceField.ElevationScalar])
    else:
      scalel=abs(f.header[0][segyio.TraceField.ElevationScalar])
#
    if int(f.header[0][segyio.TraceField.SourceGroupScalar]) != 1: 
      scalco=abs(1./f.header[0][segyio.TraceField.SourceGroupScalar])
    else:
      scalco=abs(f.header[0][segyio.TraceField.SourceGroupScalar])

    shots=[] # empty list
    src_coordinates = np.empty((len(records), 2)) # numpy array for source coordinates
    num_traces= np.empty(len(records), dtype = np.int32) # numpy array of of 'int32' type
    rec_coordinates = [] # empty list
    
    #gx=np.array([f.attributes(segyio.TraceField.GroupX),f.attributes(segyio.TraceField.GroupY)]).T
    #print(gx.shape,type(gx))
    
    #reclst=f.attributes(segyio.TraceField.GroupX)
    #idx_xrec = np.where(reclst[:] >= 0)[0]
    #reclst=f.attributes(segyio.TraceField.GroupX)[idx_xrec]
    #print(len(reclst),type(reclst))
    #for x in range(800): 
    #    print (reclst[x])

    #print(type(records),records.values())
    futures = []
    for (index, value) in enumerate(records.values()):
      #print(index,value,value.key)
      num_traces[index]=0
      rec_coordinates.append([])
      for header in value.header:
        rec_coordinates[index].append((header[segyio.TraceField.GroupX]*scalco,header[segyio.TraceField.GroupY]*scalel)) # list w receivers coordinates
        num_traces[index]+=1
      src_coordinates[index,:] = [header[segyio.TraceField.SourceX]*scalco,header[segyio.TraceField.SourceY]*scalel] # set src coordinates 
      rec_coordinates[index]=np.array(rec_coordinates[index]) #convert list to numpy array
      lst =list(records[value.key].trace)
      from_group = np.stack(lst) #numpy array with traces of the same record
      #from_group = np.stack(records[value.key].trace) #numpy array with traces of the same record
      shots.append(from_group) # append shot to list
      #futures.append(c.submit(forward_modeling_single_shot, par_files, src_coordinates[index,:], rec_coordinates[index], from_group))
      
    # Wait for all workers to finish and collect shots
    #wait(futures)
    #for i in range(len(shots)):
    #    result=futures[i].result()
    #    resampled=result.resample(num=1151)
    #    print(resampled.shape)
    #    name='shot'
    #    form = '{}_{}_dask.bin'.format(name,str(i).zfill(4))
    #    g= open(form, 'wb')
    #    np.transpose(resampled.data).astype('float32').tofile(g)

    print("Shot loading took {}".format(time.time() - start), "seconds")
    print("Number of shots {}".format(len(shots)))
    print('Size of shot file: {0} == {1}'.format(len(shots)*800.*1151.*from_group.itemsize,humanbytes(len(shots)*800.*1151.*from_group.itemsize)))

    for i, val in enumerate(rec_coordinates):
        if not np.any(val[:,0] < 0):
            break    # break here
            
    #print('-----rec_coords-----')
    #print(rec_coordinates[10][:])
    #print('--------------------')
    #print(len(rec_coordinates[10][:]))
    #remove_out_of_bounds_receivers((0.,0.), rec_coordinates[10][:])
    #print(len(rec_coordinates[10][:]))
    #print('-----rec_coords-----')
    #print(rec_coordinates[10][:])
    #print('--xxxxxxxxxxxxxxxx--')
    #sys.exit(0)
        
#only shots with all positive receivers coordinates
    #rightborder=math.ceil(src_coordinates[-1][0]/(6.25*3))+5
    #print("rightmost sample position: {0}".format(rightborder))
    #print("first shot with all receiver inside model: {0}".format(i))
#test only 5 shots
    rec_coordinates=rec_coordinates[i:i+5]
    src_coordinates=src_coordinates[i:i+5]
    shots=shots[i:i+5] 

    forward_modeling_multi_shots(c, par_files, src_coordinates,rec_coordinates,shots)

def ImagingOperator(geometry, image):
    save=True
    time_order=2
    space_order=24
    stagg_u = stagg_v = None
    x
    u = TimeFunction(name='u', grid=geometry.model.grid, staggered=stagg_u,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)
    v = TimeFunction(name='v', grid=geometry.model.grid, staggered=stagg_v,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)
    
    uu = TimeFunction(name='uu', grid=geometry.model.grid, staggered=stagg_u,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)
    vv = TimeFunction(name='vv', grid=geometry.model.grid, staggered=stagg_v,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)

    residual = PointSource(name='residual', grid=geometry.model.grid,
                           time_range=geometry.time_axis,
                           coordinates=geometry.rec_positions) 

    stencils= kernel_centered_2d(geometry.model, uu, vv, space_order, forward=False)

    stencils += residual.inject(field=uu, expr=residual * dt**2 / geometry.model.m)
    stencils += residual.inject(field=vv, expr=residual * dt**2 / geometry.model.m)
    
    # Correlate u and v for the current time step and add it to the image
    image_update = Eq(image, image - (u.dt2* uu + v.dt2*vv))
    return Operator(stencils + [image_update], subs=model.spacing_map)

# Serial modeling function
#def forward_modeling_single_shot(par_files,src_coord,rec_coord,shot):
def forward_modeling_single_shot(par_files, dict_data):

    shot_id=next(iter(dict_data))
    src_coord=dict_data[shot_id]['src']
    rec_coord=dict_data[shot_id]['rec']
    shot=dict_data[shot_id]['shot']
    
    worker = get_worker()  # The worker on which this task is running
    #for x in dict_data:
    #    print (worker.address, ':', x)
    #    for y in dict_data[x]:
    #        print (y,':',dict_data[x][y])

    # Read velocity model
    f = segyio.open(par_files[-1], iline=segyio.tracefield.TraceField.FieldRecord, 
                xline=segyio.tracefield.TraceField.CDP)
    xl, il, t = f.xlines, f.ilines, f.samples
    dz = t[1] - t[0]
    dx=f.header[1][segyio.TraceField.SourceX]-f.header[0][segyio.TraceField.SourceX]

    #print(f.bin[segyio.BinField.Format]) 4 Byte IBM Floating Point

    if len(il) != 1: 
      dims=(len(xl), len(il), len(f.samples)) 
    else:
      dims=(len(xl), len(f.samples)) 

    vp=f.trace.raw[:].reshape(dims)
    vp*=1./1000
    epsilon = np.empty(dims)
    delta = np.empty(dims)
    theta = np.empty(dims)
    params=[epsilon, delta, theta]

    # Read Thomsem parameters
    for segyfile, par in zip(par_files, params):
        f = segyio.open(segyfile, iline=segyio.tracefield.TraceField.FieldRecord, 
                xline=segyio.tracefield.TraceField.CDP)
        par[:]=f.trace.raw[:].reshape(dims)
	
    theta*=(np.pi/180.)
    print("3D velocity model dims: {0}".format(vp.shape))
    
    origin=(0.,0.)
    shape=vp.shape
    spacing=(dz,dz)
    
    #Only keep receivers within the model'
    xmin = origin[0]
    idx_xrec = np.where(rec_coord[:,0] < xmin)[0]
    is_empty = idx_xrec.size == 0
    if not is_empty:
       rec_coord=np.delete(rec_coord, idx_xrec, axis=0)

    # For 3D shot records, scan also y-receivers
    if len(origin) == 3:
        ymin = origin[1]
        idx_yrec = np.where(rec_coord[:,1] < ymin)[0]
        is_empty = idx_yrec.size == 0
        if not is_empty:
           rec_coord=np.delete(rec_coord, idx_yrec, axis=0)
           
    if rec_coord.size == 0:
        print('all receivers outside of model')
        return
        
    print('before',params[0].shape)    
    model=limit_model_to_receiver_area(rec_coord,src_coord,origin,spacing,shape,vp,params)
    print(model.vp.shape)
    #model.smooth(('epsilon', 'delta', 'theta'))

    # Geometry for current shot
    geometry = AcquisitionGeometry(model, rec_coord, src_coord, 0, 9200, f0=0.018, src_type='Ricker')
    print("Number of samples modelled data {}".format(geometry.nt))
    
    # Set up solver.
    solver_tti = AnisotropicWaveSolver(model, geometry)
    
    # Generate synthetic receiver data 
    d_calc, u, v, _ = solver_tti.forward()
    print(d_calc.data[:].shape)
    
    # Create image symbol and instantiate the previously defined imaging operator
    #image = Function(name='image', grid=model.grid)
    #op_imaging = ImagingOperator(geometry, image)
    
    #vv = TimeFunction(name='vv', grid=model.grid, staggered=None,time_order=2, space_order=space_order)
    #uu = TimeFunction(name='uu', grid=model.grid, staggered=None,time_order=2, space_order=space_order)

    #time_range = TimeAxis(start=0, stop=9200, step=8)
    #d_obs=Receiver(name='dobs', grid=model.grid,time_range=time_range,coordinates=geometry.rec_positions)
    #dobs.data[:]=shot[:].T
    #dobs_resam=dobs.resample(dt=model.critical_dt)
    
    #op_imaging(u=u, v=v,vv=vv, uu=uu, vp=model.vp, dt=model.critical_dt, 
    #           residual=dobs_resam)
    
    # Convert to numpy array and remove absorbing boundaries
    #image_crop = np.array(image.data[:])[geometry.model.nbl:-geometry.model.nbl,
    #    geometry.model.nbl:-geometry.model.nbl]

    #return leftmost,rightmost, image_crop
    return d_calc

# Parallel modeling function
def forward_modeling_multi_shots(c,par_files,sources,receivers,shots):

    maindict = dict([x, {"src":sources[x][:], "rec":receivers[x][:], "shot":shots[x][:]}] for x in range(len(shots)))
    dic_scattered = c.scatter(maindict)
    futures = []
    futures.append(c.submit(forward_modeling_single_shot, par_files, dic_scattered))
    
    #for i, src in enumerate(sources): 
    #    print(maindict[i]['src'])
    #    src_coord = src
    #    rec_coord = receivers[i]
    #    shot_record=shots[i]
    #    print('Size of shot record: {0} == {1}'.format(shot_record.size*shot_record.itemsize,humanbytes(shot_record.size*shot_record.itemsize)))
        #print(src_coord[0],rec_coord[-1][0])
        
        # Call serial modeling function for each index
        #futures.append(c.submit(forward_modeling_single_shot, par_files, rightborder, src_coord, rec_coord, shot_record))
    #    futures.append(c.submit(forward_modeling_single_shot, par_files, src_coord, rec_coord, shot_record))
        
    # Wait for all workers to finish and collect shots
    wait(futures)
    for i in range(len(shots)):
        result=futures[i].result()
        resampled=result.resample(num=1151)
        print(resampled.shape)
        name='shot'
        form = '{}_{}_dask.bin'.format(name,str(i).zfill(4))
        g= open(form, 'wb')
        np.transpose(resampled.data).astype('float32').tofile(g)
    #final_image = np.zeros((1336, 900))
    #for i in range(len(shots)):
    #    left=futures[i].result()[0]
    #    right=futures[i].result()[1]
    #    final_image[left:right,:] += futures[i].result()[2]

    #return final_image

def find_mem_pos(header_lines):
    for i,line in enumerate(header_lines):
        if('--mem=' in line):
            return i

def get_slurm_dask_client(n_workers, n_cores, n_processes):

    cluster = SLURMCluster(cores=n_cores,
                           processes=n_processes,
                           memory='20GB',
                           project="abc",
                           walltime="120000",
                           interface='ib0',
                           queue='standard')

    cluster.scale(n_workers)
    client = Client(cluster)

    return client


if __name__ == "__main__":
    print('start')
# Start Dask cluster
    cluster = LocalCluster(n_workers=4, death_timeout=600)
    c = Client(cluster)
    print('OK')
    main(c)
    #map = c.map
    #pop,log,_ =main(True)
    #plot_deap(pop, log)
    #a,b,c = main()
    #print("\n")
    #print(a)
    #print(c)

