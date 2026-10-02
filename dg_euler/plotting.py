"""Static diagnostics and fixed-scale density/vorticity movie export."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dg-euler-matplotlib')
import json,shutil,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.animation import FFMpegWriter,PillowWriter


def diagnostic_plots(histories,out):
    out=Path(out);fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for n,h in histories.items():
        t=[d['time'] for d in h];label=f'{n}²'
        axes[0,0].semilogy(t,[d['mode_amplitude'] for d in h],label=label)
        axes[0,1].plot(t,[d['transverse_kinetic'] for d in h],label=label)
        axes[1,0].semilogy(t,np.maximum([d['drift'] for d in h],1e-18),label=label)
        axes[1,1].plot(t,[d['min_pressure'] for d in h],label=label)
    for ax,title in zip(axes.flat,['Seeded Fourier amplitude A₂','Transverse kinetic energy','Normalized conserved-total drift','Minimum sampled pressure']):
        ax.set(xlabel='Time',title=title);ax.grid(alpha=.25);ax.legend()
    fig.savefig(out/'diagnostics.png',dpi=160);plt.close(fig)
    means={n:np.load(out/f'n{n}'/'means_0150.npz')['means'][...,0] for n in histories}
    low=min(float(a.min()) for a in means.values());high=max(float(a.max()) for a in means.values())
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,(n,a) in zip(axes,means.items()):
        im=ax.imshow(a,origin='lower',extent=(0,1,0,1),vmin=low,vmax=high,cmap='viridis',interpolation='nearest');ax.set(title=f'{n}² cells',xlabel='x',ylabel='y')
    fig.colorbar(im,ax=list(axes),shrink=.8,label='Cell-averaged density');fig.suptitle('Resolution comparison at t=3');fig.savefig(out/'density_comparison.png',dpi=160);plt.close(fig)
    vd=out/'validation'
    if (vd/'smooth.json').exists():
        rows=json.loads((vd/'smooth.json').read_text());fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
        for p in (1,2):
            group=[r for r in rows if r['degree']==p];ax.loglog([r['cells'] for r in group],[r['l2'] for r in group],'o-',label=f'p={p}')
        ax.set(xlabel='Cells per axis',ylabel='Density L² error',title='Smooth density-wave convergence');ax.legend();ax.grid(alpha=.3);fig.savefig(vd/'smooth.png',dpi=160);plt.close(fig)


def movie(directory='outputs/khi/n64',fps=20):
    directory=Path(directory);files=sorted((directory/'frames').glob('*.npz'))
    if not files:raise ValueError('no reconstructed movie fields found')
    cfg=json.loads((directory/'config.json').read_text());rmin=np.inf;rmax=-np.inf;omax=0.
    for file in files:
        with np.load(file) as f:
            rmin=min(rmin,float(f['density'].min()));rmax=max(rmax,float(f['density'].max()));omax=max(omax,float(abs(f['vorticity']).max()))
    norm=SymLogNorm(linthresh=max(1.,omax*.01),vmin=-omax,vmax=omax,base=10)
    fig,axes=plt.subplots(1,2,figsize=(11.2,5),constrained_layout=True)
    first=np.load(files[0]);density=axes[0].imshow(first['density'],origin='lower',extent=(0,1,0,1),vmin=rmin,vmax=rmax,cmap='viridis',interpolation='nearest')
    vort=axes[1].imshow(first['vorticity'],origin='lower',extent=(0,1,0,1),norm=norm,cmap='RdBu_r',interpolation='nearest')
    axes[0].set_title('Density');axes[1].set_title('Within-cell signed vorticity')
    for ax in axes:ax.set(xlabel='x',ylabel='y')
    fig.colorbar(density,ax=axes[0],shrink=.8);fig.colorbar(vort,ax=axes[1],shrink=.8)
    title=fig.suptitle('')
    def update(index):
        with np.load(files[index]) as f:
            density.set_data(f['density']);vort.set_data(f['vorticity'])
            title.set_text(f'Tanh double-shear KHI   t={float(f["time"]):.2f}   {cfg["nx"]}×{cfg["ny"]}, p={cfg["degree"]}   drift={float(f["drift"]):.1e}')
    ffmpeg=shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg';matplotlib.rcParams['animation.ffmpeg_path']=ffmpeg
    mp4=directory/'khi.mp4';gif=directory/'khi.gif'
    writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=2400,extra_args=['-pix_fmt','yuv420p'])
    with writer.saving(fig,str(mp4),dpi=100):
        for index in range(len(files)):
            update(index);writer.grab_frame()
    # FFmpeg palette conversion keeps all GIF frames without storing figure rasters in Python.
    subprocess.run([ffmpeg,'-y','-i',str(mp4),'-filter_complex',f'fps={fps},scale=720:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse','-loop','0',str(gif)],check=True,capture_output=True)
    for index,label in [(0,'initial'),(30,'growth'),(75,'rollup'),(len(files)-1,'final')]:
        update(min(index,len(files)-1));fig.savefig(directory/f'frame_{label}.png',dpi=130)
    plt.close(fig)
    ffprobe=shutil.which('ffprobe') or '/opt/homebrew/bin/ffprobe'
    data=json.loads(subprocess.check_output([ffprobe,'-v','error','-count_frames','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration:format=duration','-of','json',str(mp4)]))
    stream=data['streams'][0]
    if int(stream['nb_read_frames'])!=len(files) or abs(float(data['format']['duration'])-len(files)/fps)>.06:raise RuntimeError('movie metadata mismatch')
    from PIL import Image
    with Image.open(gif) as im:
        gif_frames=im.n_frames;gif_duration=sum((im.seek(i),im.info.get('duration',0))[1] for i in range(im.n_frames))/1000
    if gif_frames!=len(files) or abs(gif_duration-len(files)/fps)>.06:raise RuntimeError('GIF metadata mismatch')
    metadata=dict(mp4=data,gif=dict(frames=gif_frames,duration=gif_duration),density_limits=[rmin,rmax],vorticity_max=omax,vorticity_linthresh=max(1.,omax*.01),frames=len(files),fps=fps)
    (directory/'movie_metadata.json').write_text(json.dumps(metadata,indent=2));return metadata
