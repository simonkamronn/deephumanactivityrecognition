import subprocess


def create(folder, rate=5):
    cmdline = [
        'avconv',
        '-f',
        'image2',
        '-r',
        '%f' % rate,
        '-i',
        folder + '/custom_eval_plot_%04d.png',
        '-y',
        folder + '/movie.mp4',
    ]
    subprocess.call(cmdline)
