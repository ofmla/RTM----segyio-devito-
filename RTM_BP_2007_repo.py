import numpy as np
import segyio
import time
import math
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster, wait
from dask.distributed import as_completed, get_worker, progress
from distributed.worker import logger

from devito import *
from examples.seismic import SeismicModel, AcquisitionGeometry, TimeAxis
from examples.seismic import Receiver, PointSource
from examples.seismic.tti import AnisotropicWaveSolver
from examples.seismic.tti.operators import kernel_centered_2d
from examples.seismic.tti.operators import Gxx_centered_2d, Gzz_centered_2d
from examples.checkpointing.checkpoint import DevitoCheckpoint
from examples.checkpointing.checkpoint import CheckpointOperator
from pyrevolve import Revolver

def humanbytes(B):
    'Return the given bytes as a human friendly KB, MB, GB, or TB string'
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B/KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B/MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B/GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B/TB)


def make_lookup_table(sgy_file):
    '''
    Make a lookup of shots, where the keys are the shot record IDs being
    searched (looked up)
    '''
    lookup_table = {}
    with segyio.open(sgy_file, ignore_geometry=True) as f:
        index = None
        pos_in_file = 0
        for header in f.header:
            if int(header[segyio.TraceField.SourceGroupScalar]) < 0:
                scalco = abs(1./header[segyio.TraceField.SourceGroupScalar])
            else:
                scalco = header[segyio.TraceField.SourceGroupScalar]
            if int(header[segyio.TraceField.ElevationScalar]) < 0:
                scalel = abs(1./header[segyio.TraceField.ElevationScalar])
            else:
                scalel = header[segyio.TraceField.ElevationScalar]
    # Check to see if we're in a new shot
            if index != header[segyio.TraceField.FieldRecord]:
                index = header[segyio.TraceField.FieldRecord]
                lookup_table[index] = {}
                lookup_table[index]['filename'] = sgy_file
                lookup_table[index]['Trace_Position'] = pos_in_file
                lookup_table[index]['Num_Traces'] = 1
                lookup_table[index]['Source'] = (header[segyio.TraceField.SourceX]*scalco, header[segyio.TraceField.SourceY]*scalel)
                lookup_table[index]['Receivers'] = []
            else:  # Not in a new shot, so increase the number of traces in the shot by 1
                lookup_table[index]['Num_Traces'] += 1
            lookup_table[index]['Receivers'].append((header[segyio.TraceField.GroupX]*scalco, header[segyio.TraceField.GroupY]*scalel))
            pos_in_file += 1

    return lookup_table


def extend_image(origin, vp, model, image):
    "Extend image back to full model size"
    ndim = len(origin)
    full_image = np.zeros(vp.shape)
    nx_start = math.trunc(((model.origin[0] - origin[0])/model.spacing[0]))
    nx_end = nx_start + model.vp.shape[0]-2*model.nbl
    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(ndim))
    if ndim == 2:
        full_image[nx_start:nx_end, :] = image.data[slices]
    else:
        ny_start = math.trunc(((model.origin[1] - origin[1])/model.spacing[1]) + 1)
        ny_end = ny_start + model.vp.shape[1]
        full_gradient[nx_start:nx_end, ny_start:ny_end, :] = image.data[slices]

    return full_image


def limit_model_to_receiver_area(rec_coord, src_coord, origin, spacing, shape,
                                 vel, par, space_order=8, nbl=40, buffer=0.):
    '''
    Restrict full velocity model to area that contains either sources and
    receivers
    '''
    ndim = len(origin)

    # scan for minimum and maximum x and y source/receiver coordinates
    min_x = min(src_coord[0][0], np.amin(rec_coord[:, 0]))
    max_x = max(src_coord[0][0], np.amax(rec_coord[:, 0]))
    if ndim == 3:
        min_y = min(src_coord[0][1], np.amin(rec_coord[:, 1]))
        max_y = max(src_coord[0][1], np.amax(rec_coord[:, 0]))

    # add buffer zone if possible
    min_x = max(origin[0], min_x-buffer)
    max_x = min(origin[0] + spacing[0]*(shape[0]-1), max_x+buffer)
    if ndim == 3:
        min_y = max(origin[1], min_y-buffer)
        max_y = min(origin[1] + spacing[1]*(shape[1]-1), max_y+buffer)

    # extract part of the model that contains sources/receivers
    nx_min = int(round((min_x - origin[0])/spacing[0]))
    nx_max = int(round((max_x - origin[0])/spacing[0])+1)

    if ndim == 2:
        ox = float((nx_min)*spacing[0])
        oz = origin[-1]
    else:
        ny_min = round(min_y/spacing[1])
        ny_max = round(max_y/spacing[1])+1
        ox = float((nx_min)*spacing[0])
        oy = float((ny_min)*spacing[1])
        oz = origin[-1]

    # Extract relevant model part from full domain
    if ndim == 2:
        vel = vel[nx_min: nx_max, :]
        delta = par[0][nx_min: nx_max, :]
        epsilon = par[1][nx_min: nx_max, :]
        theta = par[2][nx_min: nx_max, :]
        phi = None
        origin = (ox, oz)
    else:
        vel = vel[nx_min:nx_max, ny_min:ny_max, :]
        delta = par[0][nx_min:nx_max, ny_min:ny_max, :]
        epsilon = par[1][nx_min:nx_max, ny_min:ny_max, :]
        theta = par[2][nx_min:nx_max, ny_min:ny_max, :]
        phi = par[3][nx_min:nx_max, ny_min:ny_max, :]
        origin = (ox, oy, oz)

    return SeismicModel(vp=vel, origin=origin, shape=vel.shape, spacing=spacing,
                 space_order=space_order, nbl=nbl, epsilon=epsilon,
                 delta=delta, theta=theta, phi=phi, bcs="damp", dtype=np.float32)


def ImagingOperator(geometry, image, space_order, save=True):

    stagg_u = stagg_v = None

    u = TimeFunction(name='u', grid=geometry.model.grid, staggered=stagg_u,
                     save=geometry.nt if save
                     else None, time_order=2, space_order=space_order)
    v = TimeFunction(name='v', grid=geometry.model.grid, staggered=stagg_v,
                     save=geometry.nt if save
                     else None, time_order=2, space_order=space_order)

    uu = TimeFunction(name='uu', grid=geometry.model.grid, staggered=stagg_u,
                      save=None, time_order=2, space_order=space_order)
    vv = TimeFunction(name='vv', grid=geometry.model.grid, staggered=stagg_v,
                      save=None, time_order=2, space_order=space_order)

    dt = geometry.dt
    residual = PointSource(name='residual', grid=geometry.model.grid,
                           time_range=geometry.time_axis,
                           coordinates=geometry.rec_positions)

    stencils = kernel_centered_2d(geometry.model, uu, vv, space_order, forward=False)

    stencils += residual.inject(field=uu.backward, expr=residual * dt**2 / geometry.model.m)
    stencils += residual.inject(field=vv.backward, expr=residual * dt**2 / geometry.model.m)

    # Correlate u and v for the current time step and add it to the image
    image_update = Eq(image, image - (u.dt2*uu + v.dt2*vv))

    return Operator(stencils + [image_update], subs=geometry.model.spacing_map)


def forward_modeling_single_shot(record, table, par_files):
    "Serial modeling function"

    worker = get_worker()  # The worker on which this task is running
    strng = '{} : {} =>'.format(worker.address, str(record).zfill(5))
    filename = 'logfile_{}.txt'.format(str(record).zfill(5))
    g = open(filename, 'w')
    g.write("This will show up in the worker logs")

    # Read velocity model
    f = segyio.open(par_files[-1], iline=segyio.tracefield.TraceField.FieldRecord,
                    xline=segyio.tracefield.TraceField.CDP)
    xl, il, t = f.xlines, f.ilines, f.samples
    dz = t[1] - t[0]
    dx = f.header[1][segyio.TraceField.SourceX]-f.header[0][segyio.TraceField.SourceX]

    if len(il) != 1:
        dims = (len(xl), len(il), len(f.samples))
    else:
        dims = (len(xl), len(f.samples))

    vp = f.trace.raw[:].reshape(dims)
    vp *= 1./1000  # convert to km/sec
    epsilon = np.empty(dims)
    delta = np.empty(dims)
    theta = np.empty(dims)
    params = [epsilon, delta, theta]

    # Read Thomsem parameters
    for segyfile, par in zip(par_files, params):
        f = segyio.open(segyfile, iline=segyio.tracefield.TraceField.FieldRecord,
                        xline=segyio.tracefield.TraceField.CDP)
        par[:] = f.trace.raw[:].reshape(dims)

    theta *= (np.pi/180.)  # use radians
    g.write('{} Parameter model dims: {}\n'.format(strng, vp.shape))

    origin = (0., 0.)
    shape = vp.shape
    spacing = (dz, dz)

    # Get a single shot as a numpy array
    filename = table[record]['filename']
    position = table[record]['Trace_Position']
    traces_in_shot = table[record]['Num_Traces']
    src_coord = np.array(table[record]['Source']).reshape((1, len(dims)))
    rec_coord = np.array(table[record]['Receivers'])

    start = time.time()
    f = segyio.open(filename, ignore_geometry=True)
    num_samples = len(f.samples)
    samp_int = f.bin[segyio.BinField.Interval]/1000
    retrieved_shot = np.zeros((traces_in_shot, num_samples))
    shot_traces = f.trace[position:position+traces_in_shot]
    for i, trace in enumerate(shot_traces):
        retrieved_shot[i] = trace
    g.write('{} Shot loaded in: {} seconds\n'.format(strng, time.time()-start))

    # Only keep receivers within the model'
    xmin = origin[0]
    idx_xrec = np.where(rec_coord[:, 0] < xmin)[0]
    is_empty = idx_xrec.size == 0
    if not is_empty:
        g.write('{} in {}\n'.format(strng, rec_coord.shape))
        idx_tr = np.where(rec_coord[:, 0] >= xmin)[0]
        rec_coord = np.delete(rec_coord, idx_xrec, axis=0)

    # For 3D shot records, scan also y-receivers
    if len(origin) == 3:
        ymin = origin[1]
        idx_yrec = np.where(rec_coord[:, 1] < ymin)[0]
        is_empty = idx_yrec.size == 0
        if not is_empty:
            rec_coord = np.delete(rec_coord, idx_yrec, axis=0)

    if rec_coord.size == 0:
        g.write('all receivers outside of model\n')
        return np.zeros(vp.shape)

    space_order = 8
    g.write('{} before: {} {} {}\n'.format(strng, params[0].shape, rec_coord.shape, src_coord.shape))
    model = limit_model_to_receiver_area(rec_coord, src_coord, origin, spacing,
                                         shape, vp, params, space_order=space_order, nbl=80)
    g.write('{} shape_vp: {}\n'.format(strng, model.vp.shape))
    model.smooth(('epsilon', 'delta', 'theta'))

    # Geometry for current shot
    geometry = AcquisitionGeometry(model, rec_coord, src_coord, 0, (num_samples-1)*samp_int, f0=0.018, src_type='Ricker')
    g.write("{} Number of samples modelled data & dt: {} & {}\n".format(strng, geometry.nt, model.critical_dt))
    g.write("{} Samples & dt: {} & {}\n".format(strng, num_samples, samp_int))

    # Set up solver.
    solver_tti = AnisotropicWaveSolver(model, geometry, space_order=space_order)
    # Create image symbol and instantiate the previously defined imaging operator
    image = Function(name='image', grid=model.grid)
    itemsize = np.dtype(np.float32).itemsize
    full_fld_mem = model.vp.size*itemsize*geometry.nt*2.

    checkpointing = True

    if checkpointing:
        op_imaging = ImagingOperator(geometry, image, space_order, save=False)
        n_checkpoints = 150
        ckp_fld_mem = model.vp.size*itemsize*n_checkpoints*2.
        g.write('Mem full fld: {} == {} use ckp instead\n'.format(full_fld_mem, humanbytes(full_fld_mem)))
        g.write('Number of checkpoints/timesteps: {}/{}\n'.format(n_checkpoints, geometry.nt))
        g.write('Memory saving: {}\n'.format(humanbytes(full_fld_mem-ckp_fld_mem)))

        u = TimeFunction(name='u', grid=model.grid, staggered=None,
                         time_order=2, space_order=space_order)
        v = TimeFunction(name='v', grid=model.grid, staggered=None,
                         time_order=2, space_order=space_order)

        vv = TimeFunction(name='vv', grid=model.grid, staggered=None,
                          time_order=2, space_order=space_order)
        uu = TimeFunction(name='uu', grid=model.grid, staggered=None,
                          time_order=2, space_order=space_order)

        cp = DevitoCheckpoint([u, v])
        op_fwd = solver_tti.op_fwd(save=False)
        op_fwd.cfunction
        op_imaging.cfunction
        wrap_fw = CheckpointOperator(op_fwd, src=geometry.src,
                                     u=u, v=v, vp=model.vp, epsilon=model.epsilon,
                                     delta=model.delta, theta=model.theta, dt=model.critical_dt)
        time_range = TimeAxis(start=0, stop=(num_samples-1)*samp_int, step=samp_int)
        dobs = Receiver(name='dobs', grid=model.grid, time_range=time_range, coordinates=geometry.rec_positions)
        if not is_empty:
            dobs.data[:] = retrieved_shot[idx_tr, :].T
        else:
            dobs.data[:] = retrieved_shot[:].T
        dobs_resam = dobs.resample(num=geometry.nt)
        g.write('Shape of residual: {}\n'.format(dobs_resam.data.shape))
        wrap_rev = CheckpointOperator(op_imaging, u=u, v=v, vv=vv, uu=uu, vp=model.vp,
                                      epsilon=model.epsilon, delta=model.delta, theta=model.theta,
                                      dt=model.critical_dt, residual=dobs_resam.data)
        # Run forward
        wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, dobs_resam.shape[0]-2)
        g.write('Revolver storage: {}\n'.format(humanbytes(cp.size*n_checkpoints*itemsize)))
        wrp.apply_forward()
        g.write('{} run finished\n'.format(strng))
        summary = wrp.apply_reverse()
        form = 'image_{}.bin'.format(str(record).zfill(5))
        h = open(form, 'wb')
        g.write('{}\n'.format(str(image.data.shape)))
        np.transpose(image.data).astype('float32').tofile(h)
    else:
        # For illustrative purposes, assuming that there is enough memory
        g.write('enough memory to save full fld: {} == {}\n'.format(full_fld_mem, humanbytes(full_fld_mem)))
        op_imaging = ImagingOperator(geometry, image, space_order)

        vv = TimeFunction(name='vv', grid=model.grid, staggered=None, time_order=2, space_order=space_order)
        uu = TimeFunction(name='uu', grid=model.grid, staggered=None, time_order=2, space_order=space_order)

        time_range = TimeAxis(start=0, stop=(num_samples-1)*samp_int, step=samp_int)
        dobs = Receiver(name='dobs', grid=model.grid, time_range=time_range, coordinates=geometry.rec_positions)
        if not is_empty:
            dobs.data[:] = retrieved_shot[idx_tr, :].T
        else:
            dobs.data[:] = retrieved_shot[:].T
        dobs_resam = dobs.resample(num=geometry.nt)

        u, v = solver_tti.forward(vp=model.vp, epsilon=model.epsilon, delta=model.delta,
                                  theta=model.theta, dt=model.critical_dt, save=True)[1:-1]

        op_imaging(u=u, v=v, vv=vv, uu=uu, epsilon=model.epsilon, delta=model.delta,
                   theta=model0.theta, vp=model.vp, dt=model.critical_dt, residual=dobs_resam)

    full_image = extend_image(origin, vp, model, image)

    return full_image


def forward_modeling_multi_shots(c, par_files, d):
    'Parallel modeling function'
    my_dict = c.scatter(d, broadcast=True)
    records = list(d.keys())
    print(records)
    futures = []
    for record in records:
        futures.append(c.submit(forward_modeling_single_shot, record, table=my_dict, par_files=par_files))

    # Check progress
    progress(futures)

    # Wait for all workers to finish and collect shots
    wait(futures)
    # Getting length of list
    length = len(futures)
    final_image = np.array(futures[0].result())
    i = 1

    print('\n::: start user output :::')
    print(c)
    print('::: end user output :::\n')

    # Iterating using while loop
    while i < length:
        final_image[:] += futures[i].result()
        i += 1

    return final_image


def find_mem_pos(header_lines):
    for i, line in enumerate(header_lines):
        if('--mem=' in line):
            return i


def get_slurm_dask_client(n_workers, n_cores, n_processes):

    cluster = SLURMCluster(cores=n_cores,
                           processes=n_processes,
                           memory='80GB',
                           interface='ib0',
                           queue='standard',
                           job_extra=['-e slurm-%j.err', '-o slurm-%j.out',
                                      '--time=72:00:00 --requeue'])

    header_lines = cluster.job_header.split('\n')
    mem_pos = find_mem_pos(header_lines)
    header_lines = header_lines[:mem_pos]+header_lines[mem_pos+1:]
    cluster.job_header = '\n'.join(header_lines)
    print(cluster.job_script())
    # Scale cluster to n_workers
    cluster.scale(n_workers)
    # Wait for cluster to start
    time.sleep(30)
    client = Client(cluster)
    print(client.scheduler_info())

    return client


def main(c):
    par_dir_files = 'ModelParams/'
    shot_dir_files = 'ModelShots/'
    par_files = ['Delta_Model.sgy', 'Epsilon_Model.sgy', 'Theta_Model.sgy', 'Vp_Model.sgy']
    shot_files = ['Anisotropic_FD_Model_Shots_part1.sgy', 'Anisotropic_FD_Model_Shots_part2.sgy',
                  'Anisotropic_FD_Model_Shots_part3.sgy', 'Anisotropic_FD_Model_Shots_part4.sgy']
    par_files = [par_dir_files + sub for sub in par_files]
    shot_files = [shot_dir_files + sub for sub in shot_files]

    table = {}
    for sfile in shot_files:
        table.update(make_lookup_table(sfile))

    # test only 5 shots
    l = list(range(210, 215))
    shots = {k: table[k] for k in l if k in table}

    final_image = forward_modeling_multi_shots(c, par_files, shots)
    g = open('image_rtm.bin', 'wb')
    #np.transpose(final_image).astype('float32').tofile(g)
    np.transpose(np.diff(final_image, axis=1)).astype('float32').tofile(g)
    print(final_image.shape)


if __name__ == "__main__":
    print('start')
    # Start Dask cluster
    # cluster = LocalCluster(n_workers=4, death_timeout=600)
    # c = Client(cluster)
    c = get_slurm_dask_client(9, 10, 1)
    print('OK')
    main(c)
